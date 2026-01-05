# Process All Products - Embedding Agent
# This script processes each product sequentially with auto-proceed mode

$products = @("Car", "Early", "Fraud", "Home", "Hospital", "Maid", "PersonalAccident")

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Processing $($products.Count) Products" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

foreach ($product in $products) {
    Write-Host "Starting: $product" -ForegroundColor Yellow
    Write-Host "Time: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray
    
    python embedding_agent.py --product $product --auto-proceed
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Completed: $product" -ForegroundColor Green
    } else {
        Write-Host "✗ Failed: $product (Exit Code: $LASTEXITCODE)" -ForegroundColor Red
    }
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "All Products Processed!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

