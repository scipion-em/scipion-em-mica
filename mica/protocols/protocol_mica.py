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
    Performs protein structure modeling guided by cryo-EM density and prior structural prediction.
    The protocol integrates an experimental electron-density map, a protein sequence, and an
    initial atomic model in order to generate a refined structural interpretation that is more
    consistent with the observed density while preserving biologically meaningful stereochemistry.

    AI Generated:

    Protein Modelling (ProtMICA) — User Manual
        Overview

        The Protein Modelling protocol is intended for situations in which a biological user has
        an experimental cryo-EM map together with a candidate structural model and wishes to obtain
        an improved atomic interpretation of the density. In modern cryo-EM workflows this is a
        particularly common situation when a predicted model, such as one derived from AlphaFold-like
        methods, captures the global fold correctly but still requires adaptation to the experimental
        map.

        The main objective of the protocol is to bridge the gap between prediction and experiment.
        Rather than treating the starting model as final, the protocol uses the density as a
        structural constraint and attempts to produce a model that better reflects the observed
        molecular state. This is especially useful when studying conformational rearrangements,
        domain motions, ligand-induced structural changes, or regions where prediction confidence is
        limited.

        Inputs and Biological Context

        The protocol requires three biologically complementary inputs. The first is the cryo-EM
        density map, which provides the experimental structural information. The second is the
        primary amino-acid sequence, which defines the molecular identity of the target. The third
        is an initial atomic model, which serves as the structural starting point for refinement.

        In practice, the most useful applications arise when the starting model is already broadly
        compatible with the target but does not yet fully match the density. This commonly occurs in
        medium-resolution cryo-EM studies, in comparative modeling projects, or in workflows where
        predicted structures must be reconciled with experimentally observed states.

        The protocol also makes use of map resolution and contour information. From a biological
        perspective, these parameters define how confidently density features can guide structural
        interpretation. Reasonable values are particularly important when maps contain heterogeneous
        regions, flexible loops, or variable local resolution.

        Preparation of Structural Information

        Before model refinement can be biologically meaningful, all input information must be placed
        into a coherent structural context. The protocol therefore organizes the sequence, density,
        and initial model into a common working framework.

        This preparation stage is particularly important when predicted models originate from
        external resources or when structures come from previous computational analyses rather than
        from direct experimental fitting. Ensuring consistency at this stage improves the reliability
        of downstream interpretation and reduces the risk of propagating incompatible structural
        assumptions.

        Density-Guided Docking

        A central biological step in the workflow is the placement of the candidate model into the
        cryo-EM map. Even when a predicted structure is globally accurate, its orientation and
        position relative to the density may not correspond to the experimentally observed state.

        Docking addresses this by establishing the approximate spatial relationship between model and
        map before more detailed refinement begins. This is particularly important for large
        assemblies, multidomain proteins, and complexes where the predicted structure captures local
        folds well but not the exact global arrangement observed experimentally.

        From a biological perspective, successful docking helps ensure that later refinement is
        driven by meaningful density correspondence rather than by arbitrary initial placement.

        Refinement and Model Adaptation

        Once the approximate placement is established, the protocol performs density-guided
        structural refinement. The purpose of this stage is not simply geometric adjustment but
        biologically informed model adaptation.

        In favorable cases, refinement improves side-chain placement, loop accommodation, domain
        positioning, and local stereochemical consistency while maintaining compatibility with the
        experimental density. This becomes especially valuable when the density reveals conformations
        that differ from those predicted computationally.

        For cryo-EM users, this means the resulting model is often more suitable for biological
        interpretation than the initial prediction alone. Regions involved in binding, catalytic
        function, inter-subunit contacts, or conformational switching may become more interpretable
        after refinement.

        Computational Flexibility

        The protocol supports execution in computational environments ranging from standard CPU-based
        systems to GPU-accelerated platforms. For practical cryo-EM work this allows adaptation to
        both exploratory desktop analyses and larger facility-scale processing environments.

        GPU acceleration is particularly valuable when dealing with large proteins, complex maps, or
        iterative modeling workflows. However, the biological interpretation of the result should
        always remain the same: computational speed may change, but structural reliability still
        depends primarily on the quality of the experimental map and the plausibility of the initial
        model.

        Output and Interpretation

        The final result is a refined atomic structure that represents a density-consistent model of
        the macromolecule. Biologically, this output is best understood as an experimentally guided
        structural hypothesis rather than as an automatically perfect reconstruction.

        In many applications, the refined model can be used for downstream interpretation, including
        inspection of conformational states, residue-level structural analysis, preparation of
        figures, interaction analysis, or subsequent refinement in other structural biology tools.

        As with any cryo-EM modeling workflow, users should visually inspect the final fit,
        especially in flexible regions, low-resolution areas, and interfaces where biological
        conclusions may be particularly sensitive to modeling assumptions.

        Practical Recommendations

        For most biological applications, the protocol performs best when the starting model already
        captures the correct global fold and when the density map has sufficient quality to guide
        meaningful structural adjustment. Strongly incorrect starting models or highly fragmented
        density may limit the interpretability of the result.

        When the map contains multiple conformational states, biological caution is particularly
        important. The refinement process may favor dominant density features, potentially obscuring
        minor but biologically relevant states.

        It is generally advisable to inspect not only the final global fit but also local regions of
        functional importance such as ligand-binding pockets, flexible loops, catalytic residues, or
        interfaces involved in oligomerization.

        Final Perspective

        For cryo-EM practitioners, protein modeling is not merely a computational finishing step but
        a central stage of structural interpretation. The biological value of the final model depends
        on how well it reconciles prior structural knowledge with experimental evidence.

        By combining sequence information, predicted structural priors, and cryo-EM density into a
        unified modeling workflow, this protocol provides a practical route toward experimentally
        informed atomic models that are better suited for downstream biological analysis.
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

        seqSrc = os.path.abspath(self.inputSeq.get().getFileName())
        structurePath = os.path.abspath(self.inputStructure.get().getFileName())
        volumePath = os.path.abspath(self.inputVolume.get().getFileName())

        af3Base = os.path.join(idFolder, "AF3_results")
        os.makedirs(af3Base, exist_ok=True)

        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("struct", structurePath)

        model = next(structure.get_models(), None)

        chains = list(model.get_chains())
        records = list(SeqIO.parse(seqSrc, "fasta"))
        
        shutil.copyfile(seqSrc, os.path.join(idFolder, os.path.basename(seqSrc)))
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
        Plugin.runCondaCommand(
            self,
            program="python",
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