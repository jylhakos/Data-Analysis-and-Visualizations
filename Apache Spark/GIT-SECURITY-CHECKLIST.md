# Git repository security

## 🔒 What's protected by .gitignore

### Secure - These files are properly ignored:

#### **Amazon AWS & security files (CRITICAL)**
- `*.pem`, `*.key`, `*.ppk` - SSH keys and certificates
- `.aws/` - AWS credential directory
- `terraform.tfstate*` - Terraform state (contains infrastructure secrets)
- `.terraform/` - Terraform working directory
- `*.tfvars` - Terraform variable files (may contain secrets)

#### **Large data files (Performance)**
- `2008.csv` (224MB) - Large airline dataset ✅ IGNORED
- `*.csv.bz2`, `*.csv.gz` - Compressed data files
- `*.zip`, `*.tar.gz` - Archive files
- Binary formats: `*.xlsx`, `*.pkl`, `*.h5`, `*.db`

#### **Development environments (Clean repo)**
- `jupyter/` directory (Python virtual environment) ✅ IGNORED
- `Mllib/` directory (MLlib environment) ✅ IGNORED
- `__pycache__/`, `*.pyc` - Python compiled files
- `.vscode/`, `.idea/` - IDE settings

#### **Logs & temporary Files**
- `*.log` - Application logs
- `*.tmp`, `*.backup` - Temporary files
- OS files: `.DS_Store`, `Thumbs.db`

### SAFE TO COMMIT - These small files are included:

#### **Project files**
- `README.md` - Documentation
- `AirTrafficProcessor.ipynb` - Main notebook
- `*.py` - Python scripts (verify, test scripts)
- `*.sh` - Shell deployment scripts

#### **Reference data (< 100KB)**
- `airports.csv` (2KB) - Airport reference data
- `carriers.csv` (546B) - Airline carrier codes  
- `2008_sample.csv` (94KB) - Small data sample
- `2008_testsample*.csv` (2-5KB) - Test data samples

#### **Amazon AWS deployment scripts**
- `aws-deployment/*.sh` - Deployment automation
- `aws-deployment/terraform/*.tf` - Infrastructure code
- `aws-deployment/*.md` - Documentation

## Security checklist

Before committing, always verify:

```bash
# Check what would be committed
git add -n .

# Verify large files are ignored
git check-ignore 2008.csv
git check-ignore jupyter/

# Check for sensitive files
find . -name "*.pem" -o -name "*.key" -o -name "*.tfstate"

# View file sizes
du -sh * | sort -hr
```

## ⚠️ NEVER COMMIT:

1. **AWS credentials** (`*.pem`, `.aws/`, API keys)
2. **Terraform state** (`*.tfstate` - contains infrastructure details)
3. **Large datasets** (>100MB - use git LFS or cloud storage)
4. **Virtual environments** (`jupyter/`, `venv/`)
5. **Compiled code** (`__pycache__/`, `*.pyc`)
6. **Personal settings** (`.vscode/`, editor configs)

## Fixes:

```bash
# If you accidentally staged a large file:
git reset HEAD 2008.csv

# Remove from git but keep locally:
git rm --cached 2008.csv

# Clean up untracked files:
git clean -fd
```