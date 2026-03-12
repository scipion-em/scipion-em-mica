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
from mica import MICA_DIC
from pwem.protocols import EMProtocol
from pyworkflow.object import String

from pwem.convert.atom_struct import pdbToCif


from pwchem import Plugin
from pwchem.objects import  SetOfStructROIs, StructROI
from pwchem.utils import insistentRun
from pwchem.constants import RDKIT_DIC, OPENBABEL_DIC

RDKIT, OPENBABEL = 0, 1



class ProtMICA(EMProtocol):
    """

    """
    _label = 'protein modelling'

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
        self._insertFunctionStep(self.runMicaStep)
        #self._insertFunctionStep(self.createOutputStep)

    def moveFilesStep(self):
        baseFolder = self._getPath('input')
        name, ext = os.path.splitext(os.path.basename(self.inputStructure.get().getFileName()))
        print(name)
        #mapId = self.getMapId()
        mapId = 0
        idFolder = os.path.join(baseFolder, str(mapId))
        os.makedirs(idFolder, exist_ok=True)
        self.idFolder = idFolder

        resultsFolder = os.path.join(idFolder, f"AF3_results/{name}_1")
        os.makedirs(resultsFolder, exist_ok=True)
        self.resultsFolder = resultsFolder

        map = os.path.abspath(self.inputVolume.get().getFileName())
        shutil.copy(map, idFolder)

        structure = os.path.abspath(self.inputStructure.get().getFileName())
        name, ext = os.path.splitext(os.path.basename(self.inputStructure.get().getFileName()))
        print(name)
        baseName = f"{name}_model_0"
        newName = os.path.join(resultsFolder, baseName + ".cif")

        shutil.copy(structure, newName)

    def runMicaStep(self):
        seqName = os.path.abspath(self.inputSeq.get().getFileName())
        mapFile = glob.glob(os.path.join(self.idFolder, "*.map"))[0]
        if self.useGpu:
            device = f'cuda:{self.gpuList.get()}'
        else:
            device = 'cpu'
        phenix = self.getPhenixEnv()
        pulchra = os.path.join(Plugin.getVar(MICA_DIC['home']), 'MICA/modules/pulchra304/pulchra')
        af3Folder = os.path.join(self.idFolder, "AF3_results")
        args = [
            f"-f {seqName}",
            f"-a {os.path.abspath(af3Folder)}",
            f"-m {os.path.abspath(mapFile)}",
            f"-c {self.contourLevel.get()}",
            f"-r {self.resolution.get()}",
            f"-p {pulchra}",
            f"-x {phenix}",
            f"-d {device}"
        ]
        path = os.path.join(Plugin.getVar(MICA_DIC['home']), 'MICA')
        Plugin.runCondaCommand(
            self,
            program='./MICA_pipeline.sh',
            args=" ".join(args),
            condaDic=MICA_DIC,
            cwd=path
        )

    def createOutputStep(self):
        resultsDir = os.path.abspath(self._getPath('results'))
        lmdbDir = os.path.abspath(self._getPath('lmdb'))

        pocketFiles = [f for f in os.listdir(lmdbDir) if f.startswith("pocket") and f.endswith(".lmdb")]
        pockets = [os.path.splitext(f)[0] for f in pocketFiles]

        allMolecules = set()
        pocketScoresDict = {}

        for pocket in pockets:
            pocketDir = os.path.join(resultsDir, pocket)
            scoreFile = None
            for f in os.listdir(pocketDir):
                if f.endswith(".txt"):
                    scoreFile = os.path.join(pocketDir, f)
                    break
            if not scoreFile:
                continue

            pocketScores = {}
            with open(scoreFile) as sf:
                for line in sf:
                    parts = line.strip().split("\t")
                    if len(parts) != 2:
                        continue
                    smi, score = parts
                    try:
                        pocketScores[smi] = float(score)
                        allMolecules.add(smi)
                    except ValueError:
                        print(f"Invalid score for molecule {smi} in pocket {pocket}")

            pocketScoresDict[pocket] = pocketScores

        allMolecules = sorted(allMolecules)
        allMoleculeFiles = [self.smiToFile.get(smi, smi) for smi in allMolecules]

        outputFile = os.path.join(self._getPath(), "results.csv")
        with open(outputFile, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Pocket"] + allMoleculeFiles)
            for pocket in pockets:
                row = [pocket]
                scores = pocketScoresDict.get(pocket, {})
                row += [scores.get(smi, 0.0) for smi in allMolecules]
                writer.writerow(row)

        outSet = SetOfStructROIs(filename=self._getPath('StructROIs.sqlite'))
        for pocket in self.pockets.get():
            outPock = StructROI()
            outPock.copy(pocket)
            outPock.Drugclip_file = String()
            outPock.setAttributeValue('Drugclip_file', str(outputFile))
            outSet.append(outPock)

        outSet.Drugclip_file = String()
        outSet.setAttributeValue('Drugclip_file', str(outputFile))

        outSet.buildPDBhetatmFile()
        self._defineOutputs(outputStructROIs=outSet)

    # --------------------------- INFO functions -----------------------------------
    def _summary(self):
        summary = ["Results csv written in protocols path: results.csv"]
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