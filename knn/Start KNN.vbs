Option Explicit
Dim shell, fs, folder, pythonExe
Set shell=CreateObject("WScript.Shell")
Set fs=CreateObject("Scripting.FileSystemObject")
folder=fs.GetParentFolderName(WScript.ScriptFullName)
pythonExe=fs.BuildPath(fs.GetParentFolderName(folder), ".venv\Scripts\pythonw.exe")
If Not fs.FileExists(pythonExe) Then
 MsgBox "ML/.venv is missing. Install requirements first.",48,"KNN Lab"
Else
 shell.CurrentDirectory=folder
 shell.Run """" & pythonExe & """ """ & fs.BuildPath(folder,"ml_knn.py") & """",0,False
End If
