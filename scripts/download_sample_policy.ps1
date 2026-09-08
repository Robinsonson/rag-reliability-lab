$ErrorActionPreference = "Stop"

$sourceUrl = "https://www.apple.com/compliance/pdfs/Business-Conduct-Policy.pdf"
$destination = Join-Path $PSScriptRoot "..\company_policy.pdf"
$expectedSha256 = "be481bdf584b463fcb7382629ec6abbf77727cfaa26efeab1950b263c9e6d137"

Write-Host "Downloading the public sample policy from Apple..."
Invoke-WebRequest -Uri $sourceUrl -OutFile $destination

$actualSha256 = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualSha256 -ne $expectedSha256) {
    throw "Downloaded file hash mismatch. Expected $expectedSha256 but received $actualSha256."
}

Write-Host "Verified company_policy.pdf (SHA-256: $actualSha256)"
