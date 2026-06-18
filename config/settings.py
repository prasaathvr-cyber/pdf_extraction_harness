import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
project_root = Path(__file__).parent.parent
load_dotenv(project_root / '.env')

# ── AWS / BEDROCK ──────────────────────────────────────────────
AWS_ACCESS_KEY_ID     = os.getenv('AWS_ACCESS_KEY_ID', '')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', '')
AWS_DEFAULT_REGION    = os.getenv('AWS_DEFAULT_REGION', 'us-west-2')
ANTHROPIC_VERSION     = os.getenv('ANTHROPIC_VERSION', 'bedrock-2023-05-31')
MODEL_ID              = os.getenv('MODEL_ID', 'us.anthropic.claude-sonnet-4-6')
READ_TIMEOUT          = int(os.getenv('READ_TIMEOUT', '300'))

# ── HARNESS LOOP ───────────────────────────────────────────────
HARNESS_MAX_ITERATIONS      = int(os.getenv('HARNESS_MAX_ITERATIONS', '100'))
# Increased from 50 to 100 because now we extract ALL pages (may take more iterations)
HARNESS_CONTEXT_LIMIT       = int(os.getenv('HARNESS_CONTEXT_LIMIT', '80000'))
HARNESS_COMPACTION_THRESHOLD = int(os.getenv('HARNESS_COMPACTION_THRESHOLD', '70000'))

# ── MODEL PARAMETERS ───────────────────────────────────────────
MODEL_MAX_TOKENS  = 4096
MODEL_TEMPERATURE = 0.2
MAX_RETRIES       = 3

# ── PAGE ANALYSIS (PHASE 1) ────────────────────────────────────
# Settings for the page analysis phase
ANALYSIS_SAMPLE_SIZE = int(os.getenv('ANALYSIS_SAMPLE_SIZE', '1500'))
# How many chars to sample from each page for classification
# Higher = more accurate but slower; Lower = faster but less accurate
# Default 1500 chars is good balance for most PDFs

ANALYSIS_CONFIDENCE_THRESHOLD = float(os.getenv('ANALYSIS_CONFIDENCE_THRESHOLD', '0.6'))
# Minimum confidence score for page type classification (0.0-1.0)
# Pages below this threshold get classified as "other"
# Default 0.6 is reasonable

# ── EXTRACTION PRIORITY (PHASE 2) ──────────────────────────────
# UPDATED: Settings for extraction priority planning
# NOTE: ALL pages are now always extracted. No SKIP category exists.
# Agent decides ORDER (HIGH → MEDIUM → LOW), not deletion.

EXTRACTION_ORDER_STRATEGY = os.getenv('EXTRACTION_ORDER_STRATEGY', 'standard')
# 'standard': HIGH → MEDIUM → LOW (default)
# 'methods_first': Extract methods and results before discussion
# Custom strategies can be added
# This controls how the priority plan is suggested to the model

EXTRACT_MINIMUM_PAGES = int(os.getenv('EXTRACT_MINIMUM_PAGES', '-1'))
# Minimum pages that MUST be extracted (safety check)
# -1 means "extract all pages in PDF"
# Set to a number to enforce minimum extraction even if model tries to stop early
# Default -1 (extract 100% of PDF)

EXTRACT_MAX_PAGES = int(os.getenv('EXTRACT_MAX_PAGES', '1000'))
# Hard cap on total pages to extract (cost/time control on very large PDFs)
# Default 1000 (very permissive; adjust for your use case)
# Example: 50 for strict cost control, 1000 for no limit on typical papers

ENFORCE_COMPLETE_EXTRACTION = os.getenv('ENFORCE_COMPLETE_EXTRACTION', 'true').lower() == 'true'
# If True: Harness will enforce extraction of ALL pages in PDF (no early stopping)
# If False: Harness will allow early stopping (not recommended)
# Default True (enforce 100% extraction)

# ── VISION EXTRACTION (PHASE 3) ────────────────────────────────
# Settings for the vision-based extraction phase
PDF_CHUNK_SIZE    = 5   # pages per extraction call (legacy, not used in new phases)
MIN_TEXT_LENGTH   = 50  # characters

# ── PATHS ──────────────────────────────────────────────────────
PROJECT_ROOT = project_root
OUTPUT_DIR   = PROJECT_ROOT / 'output'
SESSION_DIR  = PROJECT_ROOT / 'sessions'
LOGS_DIR     = PROJECT_ROOT / 'logs'
PROMPT_DIR   = PROJECT_ROOT / 'prompts' / 'instructions'

# Create dirs if missing
for d in [OUTPUT_DIR, SESSION_DIR, LOGS_DIR, PROMPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── LOGGING ────────────────────────────────────────────────────
LOG_LEVEL = os.getenv('LOG_LEVEL', 'DEBUG')
LOG_FILE  = str(LOGS_DIR / 'harness.log')

# ── PERMISSIONS ────────────────────────────────────────────────
PERMISSION_LEVELS = {'READ': 0, 'WORKSPACE': 1, 'FULL': 2}

TOOL_PERMISSIONS = {
    'analyze_pages_overview': 'READ',
    'get_extraction_priority': 'READ',
    'extract_text'           : 'READ',
    'extract_table'          : 'READ',
    'extract_figure'         : 'READ',
    'vision_extract_page'    : 'READ',
    'validate_json'          : 'READ',
    'write_file'             : 'WORKSPACE',
    'bash_execute'           : 'FULL',
}

DANGEROUS_BASH_COMMANDS = ['rm', 'rmdir', 'del', 'sudo', 'chmod', 'shutdown', 'kill']

# ── FEATURE FLAGS ──────────────────────────────────────────────
ENABLE_SESSION_PERSISTENCE  = True
ENABLE_CONTEXT_COMPACTION   = True
ENABLE_LIFECYCLE_HOOKS      = True
ENABLE_INTERACTIVE_APPROVAL = False
ENABLE_ADAPTIVE_EXTRACTION  = True  # Enable Phase 1 & 2 (page analysis + prioritization)
ENABLE_COMPLETE_EXTRACTION  = True  # NEW: Enforce extraction of ALL pages (no skipping)

# ── OUTPUT JSON SCHEMA ─────────────────────────────────────────
EXTRACTION_JSON_SCHEMA = {
    'type': 'object',
    'required': ['pdf', 'extraction_date', 'text', 'tables', 'figures'],
    'properties': {
        'pdf'            : {'type': 'string'},
        'extraction_date': {'type': 'string'},
        'text'           : {'type': 'array', 'items': {'type': 'string'}},
        'tables'         : {'type': 'array'},
        'figures'        : {'type': 'array'},
    }
}


def print_config():
    print("\n" + "="*60)
    print("PDF EXTRACTION HARNESS - CONFIG")
    print("="*60)
    print(f"  Region         : {AWS_DEFAULT_REGION}")
    print(f"  Model          : {MODEL_ID}")
    print(f"  Max Iterations : {HARNESS_MAX_ITERATIONS}")
    print(f"  Context Limit  : {HARNESS_CONTEXT_LIMIT} tokens")
    print(f"  Output Dir     : {OUTPUT_DIR}")
    print(f"  Sessions Dir   : {SESSION_DIR}")
    print(f"  Logs Dir       : {LOGS_DIR}")
    print(f"\n  --- Extraction Strategy ---")
    print(f"  Phase 1 Enabled           : True (Page Analysis)")
    print(f"  Analysis Sample Size      : {ANALYSIS_SAMPLE_SIZE} chars")
    print(f"  Phase 2 Enabled           : True (Priority Planning)")
    print(f"  Extraction Strategy       : {EXTRACTION_ORDER_STRATEGY}")
    print(f"  Phase 3 Enabled           : True (Complete Extraction)")
    print(f"  Complete Extraction       : {ENFORCE_COMPLETE_EXTRACTION} (ALL pages extracted)")
    print(f"  Minimum Pages to Extract  : {EXTRACT_MINIMUM_PAGES} (all if -1)")
    print(f"  Max Pages Cap             : {EXTRACT_MAX_PAGES}")
    print(f"  Coverage Target           : 100% of PDF")
    print("="*60 + "\n")


if __name__ == '__main__':
    print_config()