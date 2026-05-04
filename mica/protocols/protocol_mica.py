# **************************************************************************
# *
# * Authors:   Blanca Pueche (blanca.pueche@cnb.csis.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************
import os, csv, tempfile, shutil, glob
import pyworkflow.protocol.params as params
from Bio.PDB import PDBParser, MMCIFIO

from mica import MICA_DIC
from pwem.protocols import EMProtocol
from pyworkflow.object import String

from pwem.convert.atom_struct import pdbToCif
from pwem.objects import AtomStruct


from mica import Plugin
from pwchem.objects import  SetOfStructROIs, StructROI
from pwchem.utils import insistentRun
from pwchem.constants import RDKIT_DIC, OPENBABEL_DIC

RDKIT, OPENBABEL = 0, 1



class ProtMICA(EMProtocol):
    """

    """
    _label = 'protein modelling'
    stepsExecutionMode = params.STEPS_PARALLEL
    pulchra = os.path.join(Plugin.getVar(MICA_DIC['home']), 'MICA/modules/pulchra304/bin/linux/pulchra')

    # -------------------------- DEFINE param functions ----------------------

    def _defineParams(self, form):
        """ Define the input parameters that will be used.
        Params:
            form: this is the form to be populated with sections and params.
        """
        form.addHidden('useGpu', params.BooleanParam, default=True,
                       label="Use GPU for execution",
                       help="This protocol has both CPU and GPU implementation. Choose one.")

        form.addHidden('gpuList', params.StringParam, default='0',
                       label="Choose GPU IDs",
                       help="Comma-separated GPU devices that can be used.")

        form.addSection(label='Input')
        form.addParam('inputVolume', params.PointerParam, allowsNull=False,
                      pointerClass='Volume',
                      label="Input volume: ",
                      help='Select the electron map of the structure in MRC2014')

        form.addParam('resolution', params.FloatParam,
                      default=0.0,
                      label='Resolution: ',
                      help='Map resolution.')
        form.addParam('contourLevel', params.FloatParam,
                      default=0.0,
                      label='Contour level: ',
                      help='Map contour level.')

        form.addParam('inputSeq', params.PointerParam, allowsNull=False,
                      pointerClass="Sequence",
                      label="Input Sequence: ",
                      help="Input sequence.")

        form.addParam('inputStructure', params.PointerParam, allowsNull=False,
                      pointerClass='AtomStruct',
                      label="Input AF3 prediction: ",
                      help='Select the AF3 predicted structure.')


        form.addParallelSection(threads=4, mpi=1)

    # --------------------------- STEPS functions ------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.moveFilesStep)
        self._insertFunctionStep(self.processStructureStep)
        self._insertFunctionStep(self.dockInMapStep)
        self._insertFunctionStep(self.runStep)
        #self._insertFunctionStep(self.runMicaStep)
        self._insertFunctionStep(self.createOutputStep)

    def moveFilesStep(self):
        baseFolder = self._getPath('input')
        name, ext = os.path.splitext(os.path.basename(self.inputStructure.get().getFileName()))
        #mapId = self.getMapId()
        mapId = 0
        idFolder = os.path.join(baseFolder, str(mapId))
        os.makedirs(idFolder, exist_ok=True)
        self.idFolder = idFolder

        #resultsFolder = os.path.join(idFolder, f"AF3_results/{name}_1")
        from Bio import SeqIO
        seqSrc = os.path.abspath(self.inputSeq.get().getFileName())
        record = next(SeqIO.parse(seqSrc, "fasta"))
        seqId = record.id.split("|")[0].strip()
        seqId = seqId.split("_")[0]

        seqDst = os.path.join(idFolder, f"{seqId}.fasta")
        shutil.copy(seqSrc, seqDst)
        self.seqDst = seqDst

        resultsFolder = os.path.join(idFolder, f"AF3_results/{seqId}_1")
        os.makedirs(resultsFolder, exist_ok=True)
        self.resultsFolder = resultsFolder

        map = os.path.abspath(self.inputVolume.get().getFileName())
        shutil.copy(map, idFolder)

        structure = os.path.abspath(self.inputStructure.get().getFileName())
        name, ext = os.path.splitext(os.path.basename(self.inputStructure.get().getFileName()))
        #baseName = f"{name}_model_0"
        baseName = f"{name}_model_0"
        newName = os.path.join(resultsFolder, baseName + ".cif")

        if ext.lower() == ".pdb":
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure(name, structure)

            io = MMCIFIO()
            io.set_structure(structure)
            io.save(newName)
        else:
            shutil.copy(structure, newName)

    def runMicaStep(self):
        self.ensurePulchraExecutable()
        seqName = os.path.abspath(self.inputSeq.get().getFileName())
        mapFile = glob.glob(os.path.join(self.idFolder, "*.map"))[0]
        if self.useGpu:
            device = f'cuda:{self.gpuList.get()}'
        else:
            device = 'cpu'
        phenix = self.getPhenixEnv()
        af3Folder = os.path.join(self.idFolder, "AF3_results")
        args = [
            f"-f {seqName}",
            f"-a {os.path.abspath(af3Folder)}",
            f"-m {os.path.abspath(mapFile)}",
            f"-c {self.contourLevel.get()}",
            f"-r {self.resolution.get()}",
            f"-p {self.pulchra}",
            f"-x {phenix}",
            f"-d {device}"
        ]
        path = os.path.join(Plugin.getVar(MICA_DIC['home']), 'MICA')
        Plugin.runCondaCommand(
            self,
            program="bash",
            args=f'-c "PHENIX_TMP={os.path.join(self.idFolder, "phenix_tmp")} OMP_NUM_THREADS={self.numberOfThreads.get()} {os.path.join(path, "MICA_pipeline.sh")} {" ".join(args)}"',
            condaDic=MICA_DIC,
            cwd=path
        )

    def processStructureStep(self):
        seqName = os.path.abspath(self.seqDst)

        af3Folder = os.path.join(self.idFolder, "AF3_results")
        args = [
            f"-f {seqName}",
            f"-a {os.path.abspath(af3Folder)}"
        ]
        path = os.path.join(Plugin.getVar(MICA_DIC['home']), 'MICA')
        Plugin.runCondaCommand(
            self,
            program="python",
            args=f"{os.path.join(path, 'utils/process_AF3_results.py')} " + " ".join(args),
            condaDic=MICA_DIC,
            cwd=path
        )

        newName = '_1'
        structureFolder = os.path.join(self.idFolder, "AF3_structures")
        for name in os.listdir(structureFolder):
            old_path = os.path.join(structureFolder, name)

            if os.path.isdir(old_path) and name.endswith("_0.cif"):
                base = name[:-6]
                new_name = base + "_1"
                new_path = os.path.join(structureFolder, new_name)

                os.rename(old_path, new_path)


    def dockInMapStep(self):
        self.ensurePulchraExecutable()
        seqName = os.path.abspath(self.inputSeq.get().getFileName())
        mapFile = glob.glob(os.path.join(self.idFolder, "*.map"))[0]
        phenix = self.getPhenixEnv()
        af3Folder = os.path.join(self.idFolder, "AF3_results")
        args = [
            f"-m {os.path.abspath(mapFile)}",
            f"-c {self.contourLevel.get()}",
            f"-r {self.resolution.get()}",
            f"-f {seqName}",
            f"-a {os.path.abspath(af3Folder)}",
            f"-x {phenix}"
        ]
        path = os.path.join(Plugin.getVar(MICA_DIC['home']), 'MICA/utils')

        phenixTmp = os.path.abspath(os.path.join(self.idFolder, "phenix_tmp"))
        threads = self.numberOfThreads.get()

        cmd = (
                f'mkdir -p "{phenixTmp}" && '
                f'export PHENIX_TMP="{phenixTmp}" && '
                f'OMP_NUM_THREADS={threads} '
                f'OPENBLAS_NUM_THREADS={threads} '
                f'MKL_NUM_THREADS={threads} '
                f'NUMEXPR_NUM_THREADS={threads} '
                f'taskset -c 0-{threads - 1} '
                f'python {os.path.join(path, "dock_in_map.py")} ' + " ".join(args)
        )

        Plugin.runCondaCommand(
            self,
            program="bash",
            args=f'-c "{cmd}"',
            condaDic=MICA_DIC,
            cwd=path
        )

    def runStep(self):
        self.ensurePulchraExecutable()
        seqName = os.path.abspath(self.inputSeq.get().getFileName())
        mapFile = glob.glob(os.path.join(self.idFolder, "*.map"))[0]
        phenix = self.getPhenixEnv()
        if self.useGpu:
            device = f'cuda:{self.gpuList.get()}'
        else:
            device = 'cpu'

        af3Folder = os.path.join(self.idFolder, "AF3_results")
        args = [
            f"-m {os.path.abspath(mapFile)}",
            f"-f {seqName}",
            f"-a {os.path.abspath(af3Folder)}",
            f"-p {self.pulchra}",
            "--run_phenix",
            f"-x {phenix}",
            f"-r {self.resolution.get()}",
            f"--device {device}"
        ]
        path = os.path.join(Plugin.getVar(MICA_DIC['home']), 'MICA')

        phenixTmp = os.path.abspath(os.path.join(self.idFolder, "phenix_tmp"))
        threads = self.numberOfThreads.get()

        cmd = (
                f'mkdir -p "{phenixTmp}" && '
                f'PHENIX_TMP="{phenixTmp}" '
                f'OMP_NUM_THREADS={threads} '
                f'OPENBLAS_NUM_THREADS={threads} '
                f'MKL_NUM_THREADS={threads} '
                f'NUMEXPR_NUM_THREADS={threads} '
                f'taskset -c 0-{threads - 1} '
                f'python {os.path.join(path, "run.py")} ' + " ".join(args)
        )

        Plugin.runCondaCommand(
            self,
            program="bash",
            args=f'-c "{cmd}"',
            condaDic=MICA_DIC,
            cwd=path
        )

        outputFolder = os.path.join(path, 'output')
        if os.path.exists(outputFolder):
            shutil.move(outputFolder, self._getPath())

    def createOutputStep(self):
        resultsDir = os.path.abspath(self._getPath('output'))

        pdbFiles = glob.glob(os.path.join(resultsDir, "*.pdb"))
        if not pdbFiles:
            raise FileNotFoundError(f"No PDB file found in {resultsDir}")
        finalPdb = pdbFiles[0]

        struct = AtomStruct(filename=finalPdb)

        self._defineOutputs(
            outputAtomStruct=struct
        )

    # --------------------------- INFO functions -----------------------------------
    def _summary(self):
        summary = []
        return summary

    def _methods(self):
        methods = []
        return methods

    def _validate(self):
        validations = []
        return validations

    def _warnings(self):
        warnings = []
        return warnings

    # --------------------------- UTILS functions -----------------------------------
    def getMapId(self):
        fileName = os.path.abspath(self.inputVolume.get().getFileName())
        safeFileName = fileName.replace("'", "'\\''")

        fd, tempOut = tempfile.mkstemp()
        os.close(fd)

        scriptContent = (
            f"import mrcfile, re\n"
            f"emdb_id = None\n"
            f"with mrcfile.open(r'{safeFileName}', permissive=True) as m:\n"
            f"    labels = getattr(m.header, 'labels', []) or []\n"
            f"    for label in labels:\n"
            f"        try:\n"
            f"            s = label.decode('utf-8','ignore').strip() if isinstance(label, bytes) else str(label).strip()\n"
            f"        except Exception:\n"
            f"            continue\n"
            f"        match = re.search(r'EMD-(\\d+)', s)\n"
            f"        if match:\n"
            f"            emdb_id = match.group(1)\n"
            f"            break\n"
            f"with open(r'{tempOut}', 'w') as f:\n"
            f"    if emdb_id: f.write(emdb_id)\n"
        )

        # write temp Python script
        fd, scriptPath = tempfile.mkstemp(suffix=".py")
        with os.fdopen(fd, 'w') as f:
            f.write(scriptContent)

        # Run inside conda
        Plugin.runCondaCommand(
            self,
            args=[],
            condaDic=MICA_DIC,
            program=f"python {scriptPath}",
            cwd=self._getPath()
        )

        # Read EMDB ID from temp file
        with open(tempOut, 'r') as f:
            emdb_id = f.read().strip()

        os.remove(tempOut)
        os.remove(scriptPath)

        return emdb_id if emdb_id else 0

    def getPhenixEnv(self):
        phenixExec = shutil.which("phenix.real_space_refine")

        if phenixExec is None:
            raise Exception(
                "\nMICA requires PHENIX.\n"
                "Please install Phenix and ensure 'phenix.real_space_refine' "
                "is in your PATH.\n"
            )

        phenixRoot = os.path.dirname(os.path.dirname(phenixExec))
        phenixEnv = os.path.join(phenixRoot, "phenix_env.sh")

        return phenixEnv

    def ensurePulchraExecutable(self):
        import os
        if os.path.exists(self.pulchra):
            os.chmod(self.pulchra, 0o755)