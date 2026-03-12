# **************************************************************************
# *
# * Authors:  Blanca Pueche (blanca.pueche@cnb.csic.es)
# *
# * Biocomputing Unit, CNB-CSIC
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

import shutil

from scipion.install.funcs import InstallHelper

from pwchem import Plugin as pwchemPlugin
from .constants import *

_references = ['']


class Plugin(pwchemPlugin):
    @classmethod
    def defineBinaries(cls, env):
        cls.addMicaPackage(env)

    @classmethod
    def _defineVariables(cls):
        """ Return and write a variable in the config file.
        """
        cls._defineEmVar(MICA_DIC['home'], cls.getEnvName(MICA_DIC))

    @classmethod
    def addMicaPackage(cls, env, default=True):
        cls.checkPhenix()
        installer = InstallHelper(
            MICA_DIC['name'],
            packageHome=cls.getVar(MICA_DIC['home']),
            packageVersion=MICA_DIC['version']
        )

        installer.addCommand(
            "git clone https://github.com/jianlin-cheng/MICA.git",
            f"mica_cloned"
        ).addCommand(
            f"cd MICA && conda env create -f environment.yml -n {MICA_DIC['name']}-{MICA_DIC['version']}",
            "mica_env_created"
        ).addCommand(
            "cd MICA && "
            "curl https://zenodo.org/records/15756654/files/trained_models.tar.gz?download=1 --output trained_models.tar.gz && "
            "tar -xzvf trained_models.tar.gz && rm trained_models.tar.gz",
            "mica_models_downloaded"
        )

        installer.addPackage(
            env,
            dependencies=['git', 'wget', 'curl', 'make', 'g++'],
            default=default
        )

    @classmethod
    def checkPhenix(cls):
        phenix = shutil.which("phenix.real_space_refine")

        if phenix is not None:
            print("phenix already installed!")
        elif phenix is None:
            raise Exception(
                "\nMICA requires PHENIX.\n"
                "Please install Phenix and ensure 'phenix.real_space_refine' "
                "is in your PATH.\n"
            )
