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
from Bio.PDB import PDBParser, MMCIFIO, MMCIFParser, Select
from Bio import SeqIO

from mica import MICA_DIC
from pwem.protocols import EMProtocol
from pyworkflow.object import String

from pwem.convert.atom_struct import pdbToCif
from pwem.objects import AtomStruct

from mica import Plugin
from pwchem.objects import  SetOfStructROIs, StructROI
from pwchem.utils import insistentRun
from pwchem.constants import RDKIT_DIC, OPENBABEL_DIC
from pwem.convert import cifToPdb

RDKIT, OPENBABEL = 0, 1

class ChainSelect(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.id == self.chain_id

class ProtMICA(EMProtocol):
    """
      AI Generated:

      This protocol performs protein structure modeling from cryo-EM density maps
      using the MICA framework. It combines an experimental electron density map,
      a target amino acid sequence, and an AlphaFold3-predicted structure to
      generate a refined all-atom model fitted into the density map.

      The protocol prepares AlphaFold3 predictions, docks them into the input
      cryo-EM map, refines the structure using MICA and external modeling tools,
      and produces a final fitted atomic structure suitable for downstream
      structural biology analyses.

      Core Concepts
      -------------
      MICA:
          Deep learning-based framework for protein model building and refinement
          from cryo-EM density maps. It integrates predicted structures with
          experimental density information to improve structural accuracy.

      AlphaFold3 Predictions:
          Initial structural models used as starting templates. Individual chains
          are extracted and organized into the format required by MICA.

      Cryo-EM Density Map:
          Experimental electron density map (MRC/MAP format) providing spatial
          information used to guide model fitting and refinement.

      Docking and Refinement:
          AlphaFold-derived models are first positioned inside the density map
          and subsequently refined through MICA, Pulchra reconstruction, and
          PHENIX-based refinement procedures.

      Workflow
      --------
      1. Import the cryo-EM density map.
      2. Import the target protein sequence.
      3. Import an AlphaFold3-predicted structure.
      4. Split the AF3 model into individual chains.
      5. Organize AF3 outputs in the directory structure expected by MICA.
      6. Process AF3 predictions and sequence information.
      7. Dock predicted chains into the density map.
      8. Run MICA structure reconstruction and refinement.
      9. Perform optional PHENIX-based refinement.
      10. Generate a final all-atom protein model.

      Input
      -----
      - inputVolume:
          Cryo-EM density map in MRC/MAP format.

      - resolution:
          Experimental map resolution in Ångstroms.

      - contourLevel:
          Density threshold used during docking and model building.

      - inputSeq:
          Target protein sequence in FASTA format.

      - inputStructure:
          AlphaFold3-predicted protein structure used as an initial template.

      External Dependencies
      ---------------------
      - MICA
          Main modeling framework.

      - PHENIX
          Required for map-guided refinement. The PHENIX_HOME environment
          variable must be configured in Scipion.

      - Pulchra
          Used for backbone reconstruction and all-atom rebuilding.

      Output
      ------
      - outputAtomStruct:
          Refined all-atom protein structure fitted into the cryo-EM map,
          returned as an AtomStruct object.

      Generated Files
      ---------------
      The protocol creates intermediate directories containing:

      - Processed AlphaFold3 chain models
      - Docked chain structures
      - MICA reconstruction outputs
      - PHENIX refinement results
      - Final all-atom PDB model

      Use Cases
      ---------
      - Building atomic models from cryo-EM maps
      - Refining AlphaFold3 predictions using experimental density
      - Structural interpretation of cryo-EM reconstructions
      - Improving protein models prior to downstream analyses
        such as docking, molecular dynamics, or functional studies
      - Integrating AI-predicted structures with experimental data

      Notes
      -----
      - The protocol assumes the AlphaFold3 structure and input sequence
        correspond to the same biological target.

      - PHENIX must be installed and accessible through the PHENIX_HOME
        environment variable.

      - GPU execution is supported and recommended for large systems.

      - Multi-chain AlphaFold3 models are automatically separated and
        processed individually before reconstruction.
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
        self._insertFunctionStep(self.createOutputStep)

    def moveFilesStep(self):
        baseFolder = self._getPath('input')
        idFolder = os.path.join(baseFolder, "0")
        os.makedirs(idFolder, exist_ok=True)

        if self.inputSeq.get().getFileName() is not None:
            seqSrc = os.path.abspath(self.inputSeq.get().getFileName())
            shutil.copyfile(seqSrc, os.path.join(idFolder, os.path.basename(seqSrc)))
        else:
            seq = self.inputSeq.get().getSequence()
            name = self.inputSeq.get().getSeqName()
            seqSrc = os.path.join(idFolder, f"{name}.fasta")

            with open(seqSrc, "w") as f:
                f.write(f">{name}_1|Chains A|Unknown organism\n")
                f.write(f"{seq}\n")

        structurePath = os.path.abspath(self.inputStructure.get().getFileName())
        volumePath = os.path.abspath(self.inputVolume.get().getFileName())

        af3Base = os.path.join(idFolder, "AF3_results")
        os.makedirs(af3Base, exist_ok=True)

        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("struct", structurePath)

        model = next(structure.get_models(), None)

        chains = list(model.get_chains())
        records = list(SeqIO.parse(seqSrc, "fasta"))

        self.seqDst = seqSrc
        self.idFolder = idFolder
        shutil.copyfile(volumePath, os.path.join(idFolder, os.path.basename(volumePath)))

        io = MMCIFIO()

        for i, (record, chain) in enumerate(zip(records, chains), start=1):
            seqId = record.id.split("|")[0].strip().lower().split("_")[0].upper()
            folder = os.path.join(af3Base, f"{seqId}_{i}")
            os.makedirs(folder, exist_ok=True)
            outPath = os.path.join(folder, f"struct_chain_{i}_model_0.cif")
            io.set_structure(structure)
            io.save(outPath, ChainSelect(chain.id))

        self.resultsFolder = af3Base

    def processStructureStep(self):
        seqName = os.path.abspath(self.seqDst)
        af3Folder = os.path.join(self.idFolder, "AF3_results")
        args = [
            f"-f {seqName}",
            f"-a {os.path.abspath(af3Folder)}"
        ]
        path = os.path.join(Plugin.getVar(MICA_DIC['home']), 'MICA')
        gpu = self.gpuList.get()
        program = f"export CUDA_VISIBLE_DEVICES={gpu} && python"
        Plugin.runCondaCommand(
            self,
            program=program,
            args=f"{os.path.join(path, 'utils/process_AF3_results.py')} " + " ".join(args),
            condaDic=MICA_DIC,
            cwd=path
        )

    def dockInMapStep(self):
        self.ensurePulchraExecutable()
        seqName = os.path.abspath(self.seqDst)
        mapFiles = (
                glob.glob(os.path.join(self.idFolder, "*.map")) +
                glob.glob(os.path.join(self.idFolder, "*.mrc")) +
                glob.glob(os.path.join(self.idFolder, "*.mrc.gz"))
        )

        mapFile = mapFiles[0]
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
        gpu = self.gpuList.get()

        cmd = (
                f'mkdir -p "{phenixTmp}" && '
                f'export PHENIX_TMP="{phenixTmp}" && '
                f'OMP_NUM_THREADS={threads} '
                f'OPENBLAS_NUM_THREADS={threads} '
                f'MKL_NUM_THREADS={threads} '
                f'NUMEXPR_NUM_THREADS={threads} '
                f'taskset -c 0-{threads - 1} '
                f'export CUDA_VISIBLE_DEVICES={gpu} && '
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
        seqName = os.path.abspath(self.seqDst)
        mapFiles = (
                glob.glob(os.path.join(self.idFolder, "*.map")) +
                glob.glob(os.path.join(self.idFolder, "*.mrc")) +
                glob.glob(os.path.join(self.idFolder, "*.mrc.gz"))
        )

        mapFile = mapFiles[0]
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

        pdbFiles = glob.glob(os.path.join(resultsDir, "*all_atom_model.pdb"))
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
        phenixHome = Plugin.getVar('PHENIX_HOME', None)

        if not phenixHome:
            phenixHome = os.environ.get('PHENIX_HOME', None)

        if not phenixHome or not os.path.exists(phenixHome):
            raise Exception(
                "\nMICA requires PHENIX.\n"
                "Install phenix and define de PHENIX_HOME variable in scipion.conf.\n"
            )
        phenixEnv = os.path.join(phenixHome, "phenix_env.sh")

        if not os.path.exists(phenixEnv):
            alternativePath = os.path.join(phenixHome, "build", "phenix_env.sh")
            if os.path.exists(alternativePath):
                phenixEnv = alternativePath
            else:
                raise FileNotFoundError(
                    f"\nFound PHENIX_HOME in '{phenixHome}', but not "
                    f"'phenix_env.sh'\n"
                )
        return phenixEnv

    def ensurePulchraExecutable(self):
        import os
        if os.path.exists(self.pulchra):
            os.chmod(self.pulchra, 0o755)