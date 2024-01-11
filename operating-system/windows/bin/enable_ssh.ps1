# Install OpenSSH server
DISM.exe /Online /Add-Capability /CapabilityName:OpenSSH.Server~~~~0.0.1.0

# Validate
Get-WindowsCapability -Online | ? Name -like 'OpenSSH.Server*'

# Start the sshd service
Set-Service -Name sshd -StartupType 'Automatic'
Start-Service sshd

# Check if sshd service is running and waiting for connections on port 22
netstat -nao | find /i '":22"'

# Check if the firewall allows inbound connection to port 22
Get-NetFirewallRule -Name *OpenSSH-Server* |select Name, DisplayName, Description, Enabled

# Change the default shell for OpenSSH for Windows to PowerShell
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell -Value "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -PropertyType String -Force
