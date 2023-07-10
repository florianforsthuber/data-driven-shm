"""ABAQUS SCRIPT
Extracts the information of each node, which is stored in the .odb file of a submodel simulation result.
The information transfer between the python project and this abaqus script is realized via a JSON file.

JSON file structure:
filename: extract_node_data_support_V2.json
Dictonary with keys:
* odb_filepath: -> Path, .odb file containing simulation result
* target_dir: -> Path, directory where .csv file of extraction result should be stored
* Step: -> str, Simulation step holding the FieldOutput data
* Instance: -> str, Name of Instance to extract node data from
* NodeSet: -> str, All nodes of Instance, typically SET-27
* sP_layer: -> str, ('Bottom_ply', 'Top_ply') specifing which ply to extract nodes from in case of shell elements

This script distinguishes two types of elements by the name of the instance.
CORE -> solid elements
BOTTOM, TOP -> shell elements

1. The field outputs S (stress), E (strain) and U (displacement) of one instance are read from odb file
2. Values of the field outputs are stored with respect to the nodes -> extrapolation from integration points
3. The multiple values available at each node are averaged to ine specific node value
4. These values are written to a csv file

Author: student k1256205@students.jku.at
Created: 25/07/2022
"""
# -*- coding: utf-8 -*-
import os
from odbAccess import *
from abaqusConstants import *
import numpy as np
import csv
import fnmatch
import json

"""
Helper functions
"""

# read paths needed inside script from json file
with open("extract_node_data_support.json", "r") as read_file:
    extraction_settings = json.load(read_file)

odb_filepath, odb_filename = os.path.split(extraction_settings['odb_filepath'])

odb_filepath = str(odb_filepath)
odb_filename = str(odb_filename)
target_dir = str(extraction_settings['target_dir'])
sP_layer = str(extraction_settings['sP_layer'])
instance_name = str(extraction_settings['Instance'])
step_name = str(extraction_settings['Step'])

# new file name without suffix
instance_map = {"SUB-SKIN-BOTTOM-1": "bottom",
                "SUB-SKIN-TOP-1": "top",
                "SUB-CORE-1": "core"}
new_file_name = odb_filename.split('.')[0] + "_" + instance_map[instance_name] + "_nodal"
#change cwd to directory where new odb files are stored
os.chdir(odb_filepath)

# load obd object
odb = openOdb(path=odb_filename, readOnly=True)

# load data from obd object according to given settings
instance = odb.rootAssembly.instances[instance_name]
fieldOutputs = odb.steps[step_name].frames[-1].fieldOutputs

# programatically get nodeSet
nodeSet_list = instance.nodeSets.keys()
# select list item which is either set-27 or set-1 -> ideally .odb only one set would be defined
nodeSet_name = [n_set for n_set in nodeSet_list if n_set in ['SET-1', 'SET-27']][0]
nodeSet = instance.nodeSets[nodeSet_name]
elemSet = instance.elementSets[nodeSet_name]

# distinguish between Core (continuum solid) and Shell instances
if "CORE" in instance.name:
    # handle extraction specific to solid elements
    # solid elements to not have sectionPoints -> no composite layers etc.
    stressSet = fieldOutputs['S'].getSubset(region=instance, position=ELEMENT_NODAL)
    strainSet = fieldOutputs['E'].getSubset(region=instance, position=ELEMENT_NODAL)
    # displacement values are calculated at nodes by abaqus
    dispSet = fieldOutputs['U'].getSubset(region=instance, position=NODAL)
    dispRotSet = fieldOutputs['UR'].getSubset(region=instance, position=NODAL)
else:
    # handle extraction specific to shells -> sectionPoint, etc
    # shell elements have sections, accessible over section points (composite layers)
    # not providing a sectionPoint (determines layer) results in
    # subset of both top and bottom ply results of that shell element
    ply_to_sP = {'Top_ply': -1, 'Bottom_ply:': 0}
    sP = elemSet.elements[0].sectionCategory.sectionPoints[ply_to_sP[sP_layer]]

    # strain and stress values are calculated at integration points by abaqus -> extrapolation to nodes neccessary
    # position=ELEMENT_NODAL -> extrapolation of value at integration point to all nodes of that element
    stressSet = fieldOutputs['S'].getSubset(region=instance, position=ELEMENT_NODAL, sectionPoint=sP)
    strainSet = fieldOutputs['E'].getSubset(region=instance, position=ELEMENT_NODAL, sectionPoint=sP)
    # displacement values are calculated at nodes by abaqus
    dispSet = fieldOutputs['U'].getSubset(region=instance, position=NODAL)
    dispRotSet = fieldOutputs['UR'].getSubset(region=instance, position=NODAL)

# Get STRESS values from fieldOutput object
nStresses = len(stressSet.values)
allStresses = np.empty([nStresses, 10])
for i, v in enumerate(stressSet.values):
    allStresses[i, 0] = v.nodeLabel
    allStresses[i, 1] = v.data[0]  # S11
    allStresses[i, 2] = v.data[1]  # S22
    allStresses[i, 3] = v.data[2]  # S33
    allStresses[i, 4] = v.data[3]  # S12
    # allStresses[i,1:5] = v.data #alternative to export S11, S22, S33 and S12 in one line
    allStresses[i, 5] = v.mises
    allStresses[i, 6] = v.maxPrincipal
    allStresses[i, 7] = v.minPrincipal
    allStresses[i, 8] = v.maxInPlanePrincipal
    allStresses[i, 9] = v.minInPlanePrincipal

# Get STRAIN values from fieldOutput object
nStrains = len(strainSet.values)
allStrains = np.empty([nStrains, 10])
for i, v in enumerate(strainSet.values):
    allStrains[i, 0] = v.nodeLabel
    allStrains[i, 1] = v.data[0]  # E11
    allStrains[i, 2] = v.data[1]  # E22
    allStrains[i, 3] = v.data[2]  # E33
    allStrains[i, 4] = v.data[3]  # E12
    allStrains[i, 5] = v.mises
    allStrains[i, 6] = v.maxPrincipal
    allStrains[i, 7] = v.minPrincipal
    allStrains[i, 8] = v.maxInPlanePrincipal
    allStrains[i, 9] = v.minInPlanePrincipal

# Get DISPLACEMENT values from fieldOutput object
nDisps = len(dispSet.values)
allDisps = np.empty([nDisps, 5])
for i, v in enumerate(dispSet.values):
    allDisps[i, 0] = v.nodeLabel
    allDisps[i, 1] = v.data[0]  # U1
    allDisps[i, 2] = v.data[1]  # U2
    allDisps[i, 3] = v.data[2]  # U3
    # allDisps[i,3] = v.dataDouble[2]  #U3
    allDisps[i, 4] = v.magnitude  # U Magnitude

# Get ROTATIONAL DISPLACEMENT values from fieldOutput object
nDispsRot = len(dispRotSet.values)
allDispsRot = np.empty([nDispsRot, 5])
for i,v in enumerate(dispRotSet.values):
    allDispsRot[i, 0] = v.nodeLabel
    allDispsRot[i, 1] = v.data[0]  # UR1
    allDispsRot[i, 2] = v.data[1]  # UR2
    allDispsRot[i, 3] = v.data[2]  # UR3
    allDispsRot[i, 4] = v.magnitude  # UR_mag

# because of ELEMENT_NODAL flag, nodes are defined with multiple values, extrapolated from the neighboring
# integration points -> all the results available at one node will be averaged to one node-spcific value

# write node data to csv file
csv_filename = new_file_name + ".csv"
with open(os.path.join(target_dir, csv_filename), 'wb') as csvfile:
    csvWriter = csv.writer(csvfile, delimiter=',')
    csvWriter.writerow(['nodeLabel', 'x', 'y', 'z',
                        'U_1', 'U_2', 'U_3', 'U_mag',
                        'UR_1', 'UR_2', 'UR_3', 'UR_mag',
                        'S_11', 'S_22', 'S_33', 'S_12', 'S_mises', 'S_maxPrincipal', 'S_minPrincipal',
                        'S_maxInPlanePrincipal', 'S_minInPlanePrincipal',
                        'E_11', 'E_22', 'E_33', 'E_12', 'E_mises', 'E_maxPrincipal', 'E_minPrincipal',
                        'E_maxInPlanePrincipal', 'E_minInPlanePrincipal'])
    for n in instance.nodes:  # elements of nodeSet.nodes and instance.nodes are each equal -> checked
        # at which indices are stress results for the current node
        indicesStress = np.where(allStresses[:, 0] == float(n.label))
        # get this results and average for each stress invariant (mises, maxPrincipal)
        stressesAtIndices = allStresses[indicesStress, :]
        S_11 = np.mean(stressesAtIndices[0][:, 1])
        S_22 = np.mean(stressesAtIndices[0][:, 2])
        S_33 = np.mean(stressesAtIndices[0][:, 3])
        S_12 = np.mean(stressesAtIndices[0][:, 4])
        S_mises = np.mean(stressesAtIndices[0][:, 5])
        S_maxPrincipal = np.mean(stressesAtIndices[0][:, 6])
        S_minPrincipal = np.mean(stressesAtIndices[0][:, 7])
        S_maxInPlanePrincipal = np.mean(stressesAtIndices[0][:, 8])
        S_minInPlanePrincipal = np.mean(stressesAtIndices[0][:, 9])

        # at which indices are strain results for the current node
        indicesStrain = np.where(allStrains[:, 0] == float(n.label))
        # get this results and average to get single value for the current node
        strainsAtIndices = allStrains[indicesStrain, :]
        E_11 = np.mean(strainsAtIndices[0][:, 1])
        E_22 = np.mean(strainsAtIndices[0][:, 2])
        E_33 = np.mean(strainsAtIndices[0][:, 3])
        E_12 = np.mean(strainsAtIndices[0][:, 4])
        E_mises = np.mean(strainsAtIndices[0][:, 5])
        E_maxPrincipal = np.mean(strainsAtIndices[0][:, 6])
        E_minPrincipal = np.mean(strainsAtIndices[0][:, 7])
        E_maxInPlanePrincipal = np.mean(strainsAtIndices[0][:, 8])
        E_minInPlanePrincipal = np.mean(strainsAtIndices[0][:, 9])

        # at which indices are displacement results for the current node
        indicesDisp = np.where(allDisps[:, 0] == float(n.label))
        # averaging is actually not necessary for displacements (node results!)
        dispsAtIndices = allDisps[indicesDisp, :]
        U_1 = np.mean(dispsAtIndices[0][:, 1])
        U_2 = np.mean(dispsAtIndices[0][:, 2])
        U_3 = np.mean(dispsAtIndices[0][:, 3])
        U_mag = np.mean(dispsAtIndices[0][:, 4])

        # at which indices are rotational displacement results for the current node
        indicesDispRot = np.where(allDispsRot[:, 0] == float(n.label))
        # averaging is actually not necessary for displacements (node results!)
        dispsRotAtIndices = allDispsRot[indicesDispRot, :]
        UR_1 = np.mean(dispsRotAtIndices[0][:, 1])
        UR_2 = np.mean(dispsRotAtIndices[0][:, 2])
        UR_3 = np.mean(dispsRotAtIndices[0][:, 3])
        UR_mag = np.mean(dispsRotAtIndices[0][:, 4])

        # write row data to csv file
        csvWriter.writerow([n.label, n.coordinates[0], n.coordinates[1], n.coordinates[2],
                            U_1, U_2, U_3, U_mag,
                            UR_1, UR_2, UR_3, UR_mag,
                            S_11, S_22, S_33, S_12, S_mises, S_maxPrincipal, S_minPrincipal,
                            S_maxInPlanePrincipal, S_minInPlanePrincipal,
                            E_11, E_22, E_33, E_12, E_mises, E_maxPrincipal, E_minPrincipal,
                            E_maxInPlanePrincipal, E_minInPlanePrincipal])
odb.close()