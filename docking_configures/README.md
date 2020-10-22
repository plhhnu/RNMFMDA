SARS-Cov2 Docking experiment
=======


### Introdiction
We used the AutoDockTools for molecular docking of two important proteins: SARS-Cov-2 S Protein and human ACE2 Protein. And this folder contain the information about the detial of molecular docking.This folder contain two folders named S and ACE2, the name represent the receptor, and they all also contain two folders named by name of the ligands. These folders contain all the information for molecular docking. For example, the remdesivir folder under the ACE2 folder mean this folder the result of molecular docking between remdesivir and ACE2.

### Docking result
These folder will have some important file as follow:
|File|Function|
|--|--|
|rrr.pdbqt|protein receptor file|
|remdesivir(ribavirin).pdbqt|ligand file|
|remdesivir(ribavirin).csv|docking result file ranked by binding energy|
|remdesivir(ribavirin).gpf|grid parameter file|
|remdesivir(ribavirin).dpf|docking parameter file|
|remdesivir(ribavirin).dlg|molecular docking result|
|rank1.pdbqt|the three-dimensional structures file of the lowerst binding energy result|

You can use the AutoDockTools open the dpf file to anaysis the detail of molecular docking, and you also can use the pymol to watch the three-dimensional structures of the docking result by opening the rank1.pdbqt