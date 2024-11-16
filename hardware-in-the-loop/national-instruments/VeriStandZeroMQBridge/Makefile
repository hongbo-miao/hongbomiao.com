nuget-restore:
	nuget restore VeriStandZeroMQBridge.sln
dotnet-tool-restore:
	dotnet tool restore

lint-csharp-format:
	dotnet format VeriStandZeroMQBridge.sln --verify-no-changes
lint-csharp-format-fix:
	dotnet format VeriStandZeroMQBridge.sln
lint-csharp-csharpier:
	dotnet csharpier --check .
lint-csharp-csharpier-fix:
	dotnet csharpier .
