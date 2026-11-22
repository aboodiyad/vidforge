; VidForge installer (Inno Setup 6). Built by packaging\build.py, which passes
; AppVersion, SourceDir (the app folder), OutputDir and IconFile.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{6E0B7C35-1F0B-4C8B-9E7A-6D2C8F4A9B21}
AppName=VidForge
AppVersion={#AppVersion}
AppVerName=VidForge {#AppVersion}
AppPublisher=VidForge
AppPublisherURL=https://github.com/aboodiyad/vidforge
AppSupportURL=https://github.com/aboodiyad/vidforge/issues
DefaultDirName={localappdata}\Programs\VidForge
DisableProgramGroupPage=yes
DisableDirPage=auto
PrivilegesRequired=lowest
OutputDir={#OutputDir}
OutputBaseFilename=VidForge-Setup-{#AppVersion}
SetupIconFile={#IconFile}
UninstallDisplayIcon={app}\vidforge.ico
UninstallDisplayName=VidForge
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
WizardStyle=modern
ShowLanguageDialog=auto
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"
Name: "ar"; MessagesFile: "compiler:Languages\Arabic.isl"

[CustomMessages]
en.DeleteData=Also delete the programs and AI files VidForge downloaded?%n%nThey can take several GB. Your own videos are not touched.
ar.DeleteData=هل تريد أيضًا حذف البرامج وملفات الذكاء الاصطناعي التي نزّلها VidForge؟%n%nقد يبلغ حجمها عدة غيغابايت. فيديوهاتك لن تُمسّ.

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[InstallDelete]
; A new version replaces the whole bundled Python, so no old files stay behind. Downloads stay.
Type: filesandordirs; Name: "{app}\python"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
; Its presence tells VidForge to keep programs, models and settings here (portable mode).
Name: "{app}\vidforge-data"

[Icons]
Name: "{autoprograms}\VidForge"; Filename: "{app}\python\pythonw.exe"; Parameters: "-m vidforge.app"; WorkingDir: "{app}"; IconFilename: "{app}\vidforge.ico"; AppUserModelID: "VidForge.VidForge"
Name: "{autodesktop}\VidForge"; Filename: "{app}\python\pythonw.exe"; Parameters: "-m vidforge.app"; WorkingDir: "{app}"; IconFilename: "{app}\vidforge.ico"; AppUserModelID: "VidForge.VidForge"; Tasks: desktopicon

[Run]
Filename: "{app}\python\pythonw.exe"; Parameters: "-m vidforge.app"; WorkingDir: "{app}"; Description: "{cm:LaunchProgram,VidForge}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Files Python writes while running (bytecode caches)
Type: filesandordirs; Name: "{app}\python"

[Code]
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Data: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    Data := ExpandConstant('{app}\vidforge-data');
    if DirExists(Data) then
      if (not UninstallSilent) and (MsgBox(CustomMessage('DeleteData'), mbConfirmation, MB_YESNO) = IDYES) then
        DelTree(Data, True, True, True);
    RemoveDir(ExpandConstant('{app}'));
  end;
end;
