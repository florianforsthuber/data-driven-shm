"""ABAQUS SCRIPT to create new submodels
!! Abaqus script -> To be run by abaqus -> !! Python 2.7
Makes a copy of the current damaged submodel "Submodel-Edge-SC-Damage"
and adaptes the damage according to csv file with new
damage locations and sizes.

1. Run this script from Abaqus main model: File -> Run Script...
2. The .csv file containing the damage locations should be in the same directory as this file (/scripts)
3. The script will change the working directory of abaqus to the submodel results folder (/submodels)
4. After completion the working directory is changed back to the original path (directory of .cae model)

Documentation of the underlying script (ParameterStudy_DamageData.py):
# This script will be able to copy models with modifiying radian, the middle point of the defects.
# Porcedure of the programm:
# 1. Creating a copy of the existing models
# 2. Adaption of the defect: size and location
# 3. Remeshing the assembly
# 4. Have the constraints had to be updated?
# 5. Performing buckling simulation
# 6. Modify or supress the buckliung step.
# 7. Performing the load step
# 8. Matlkab Script with automatically extracting the strain data.
# 9. Loading the data to python again and train the models

Author: student k1256205@students.jku.at
Created: 19/06/2022
"""
from abaqus import *
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
import csv
import os
import shutil

# load .csv file as dictionary before changing working directory
damage_para = []
filename = os.path.join(os.getcwd(), 'scripts', 'Damage_locations.csv')
with open(filename) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        damage_para.append(row)

#save a copy of 'Damage_locations.csv' to the submodels directory
# to use as lookup table for Damage State and labels (x,y,r) and to document
# the basis of the .inp files
shutil.copy(filename, os.path.join(os.getcwd(), 'submodels', 'Basis_damage_locations.csv'))

# change working directory to submodel results folder '/submodels'
current_dir = os.getcwd()
os.chdir(os.path.join(current_dir, 'submodels'))

#Copy the model, a for iteration has to be implemented here, with adaption of the parameters and modelnames
ModelName = 'Part'
mdb.Model(name=ModelName,objectToCopy=mdb.models['Submodel-Edge-SC-Damage'])

# create submodel input file
for damage in damage_para[99:100]:
#for damage in damage_para
    # define unique job name (also name of the input file)
    job_name = 'submodel_' + damage['name']

    # convert to strings for setValues
    para_radi = str(damage['radius'])
    para_dis_x = str(damage['x'])
    para_dis_y = str(damage['y'])

    # Abaqus scripting routine
    p = mdb.models[ModelName].parts['Sub-Skin-Bottom']
    s = p.features['Partition face-1'].sketch
    mdb.models[ModelName].ConstrainedSketch(name='__edit__', objectToCopy=s)
    s1 = mdb.models[ModelName].sketches['__edit__']
    g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
    s1.setPrimaryObject(option=SUPERIMPOSE)
    p.projectReferencesOntoSketch(sketch=s1, upToFeature=p.features['Partition face-1'], filter=COPLANAR_EDGES)
    s = mdb.models[ModelName].sketches['__edit__']
    s.parameters['radi'].setValues(expression=para_radi)
    s.parameters['dis_x'].setValues(expression=para_dis_x)
    s.parameters['dis_y'].setValues(expression=para_dis_y)
    # s1.unsetPrimaryObject()
    p = mdb.models[ModelName].parts['Sub-Skin-Bottom']
    p.features['Partition face-1'].setValues(sketch=s1)
    p.regenerate()

    # Defining the steps,
    # 1. Supress the Load Step --> only Buckle step is active;
    # 2. Resume Load Step, Supress Buckle step, --> only Load Step step is active
    # suppressing the buckle step node output --> is inactive
    mdb.models[ModelName].steps['Step-2_Buckle'].suppress()
    mdb.models[ModelName].keywordBlock.synchVersions(storeNodesAndElements=False)

    # Meshing the model
    p = mdb.models[ModelName].parts['Sub-Core']
    p.generateMesh()
    p = mdb.models[ModelName].parts['Sub-Skin-Bottom']
    p.generateMesh()
    p = mdb.models[ModelName].parts['Sub-Skin-Top']
    p.generateMesh()

    #Creating the job -> abaqus scripting reference guide 26.4.1 Members of ModelJob object
    desc = 'Submodel, instance SUB-SKIN-BOTTOM-1: random damage state ' + damage['name']
    mdb.Job(name=job_name, model=ModelName, description=desc, type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, memoryUnits=PERCENTAGE,
    getMemoryFromAnalysis=True, explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='',
    scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=1, numGPUs=0)

    # write .inp file, named {JobName}.inp (to directory where .cae file is located, in which the script was run)
    mdb.jobs[job_name].writeInput(consistencyChecking=OFF)

# restore current working directory
os.chdir(current_dir)