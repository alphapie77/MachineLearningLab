Option Explicit
Dim shell, fs, folder, pythonExe, candidate, process, output
Set shell = CreateObject("WScript.Shell")
Set fs = CreateObject("Scripting.FileSystemObject")
folder = fs.GetParentFolderName(WScript.ScriptFullName)
pythonExe = fs.BuildPath(fs.GetParentFolderName(folder), ".venv\Scripts\pythonw.exe")
' Locate a real Python executable; the final launch uses pythonw (no console).
For Each candidate In Array("python", "py -3")
    If fs.FileExists(pythonExe) Then Exit For
    On Error Resume Next
    Set process = shell.Exec(candidate & " -c ""import sys; print(sys.executable)""")
    If Err.Number = 0 Then
        output = Trim(process.StdOut.ReadAll)
        If process.ExitCode = 0 And fs.FileExists(output) Then
            pythonExe = fs.BuildPath(fs.GetParentFolderName(output), "pythonw.exe")
        End If
    End If
    Err.Clear
    On Error GoTo 0
    If fs.FileExists(pythonExe) Then Exit For
Next
If Not fs.FileExists(pythonExe) Then
    MsgBox "Python 3 with Tkinter is required. Install Python with its Tcl/Tk option, then open this file again.", 48, "K-Means Lab"
Else
    shell.CurrentDirectory = folder
    shell.Run """" & pythonExe & """ """ & fs.BuildPath(folder, "launch_kmeans.pyw") & """", 0, False
End If
