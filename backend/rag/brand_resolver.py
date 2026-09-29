"""
Resolves brand names to generic drug names.
Priority:
  1. Local Indian brand dictionary (instant)
  2. RxNorm API (works for US brands)
  3. OpenFDA API (fallback)
  4. Return original name
"""

import requests
import time

RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"

# ── Indian brand name dictionary ───────────────────────────────
# Add more as needed — this covers the most common ones
INDIAN_BRANDS: dict[str, list[str]] = {
    # Add these to INDIAN_BRANDS dict
    "paracetamol":    ["acetaminophen"],
    "acetaminophen":  ["acetaminophen"],
    "ibuprofen":      ["ibuprofen"],
    "aspirin":        ["aspirin"],
    "metformin":      ["metformin"],
    "amoxicillin":    ["amoxicillin"],
    "montelukast":    ["montelukast"],
    "levocetirizine": ["levocetirizine"],
    # Pain / Fever
    "crocin":         ["paracetamol", "acetaminophen"],
    "calpol":         ["paracetamol", "acetaminophen"],
    "dolo":           ["paracetamol", "acetaminophen"],
    "combiflam":      ["ibuprofen", "paracetamol"],
    "brufen":         ["ibuprofen"],
    "disprin":        ["aspirin"],
    "voveran":        ["diclofenac"],
    "volini":         ["diclofenac"],

    # Allergy / Cold
    "montair":        ["montelukast"],
    "montair lc":     ["montelukast", "levocetirizine"],
    "montair-lc":     ["montelukast", "levocetirizine"],
    "telekast l":     ["montelukast", "levocetirizine"],
    "levocet":        ["levocetirizine"],
    "cetirizine":     ["cetirizine"],
    "allegra":        ["fexofenadine"],
    "nasoclear":      ["sodium chloride"],
    "sinarest":       ["paracetamol", "phenylephrine", "cetirizine"],

    # Antibiotics
    "augmentin":      ["amoxicillin", "clavulanate"],
    "mox":            ["amoxicillin"],
    "zithromax":      ["azithromycin"],
    "azithral":       ["azithromycin"],
    "cifran":         ["ciprofloxacin"],
    "taxim":          ["cefotaxime"],
    "monocef":        ["ceftriaxone"],

    # Stomach / Gastro
    "pan":            ["pantoprazole"],
    "pantocid":       ["pantoprazole"],
    "omez":           ["omeprazole"],
    "rantac":         ["ranitidine"],
    "gelusil":        ["aluminum hydroxide", "magnesium hydroxide"],
    "digene":         ["aluminum hydroxide", "magnesium hydroxide"],
    "domperidone":    ["domperidone"],
    "domstal":        ["domperidone"],
    "ondansetron":    ["ondansetron"],

    # Diabetes
    "glycomet":       ["metformin"],
    "glucophage":     ["metformin"],
    "januvia":        ["sitagliptin"],
    "galvus":         ["vildagliptin"],

    # Blood pressure / Heart
    "amlodac":        ["amlodipine"],
    "norvasc":        ["amlodipine"],
    "telma":          ["telmisartan"],
    "telmikind":      ["telmisartan"],
    "atenolol":       ["atenolol"],
    "tenormin":       ["atenolol"],
    "ecosprin":       ["aspirin"],
    "cardace":        ["ramipril"],

    # Cholesterol
    "atorlip":        ["atorvastatin"],
    "lipitor":        ["atorvastatin"],
    "rozucor":        ["rosuvastatin"],
    "crestor":        ["rosuvastatin"],

    # Thyroid
    "thyronorm":      ["levothyroxine"],
    "eltroxin":       ["levothyroxine"],

    # Vitamins / Supplements
    "shelcal":        ["calcium carbonate", "vitamin d3"],
    "becosules":      ["vitamin b complex"],
    "zincovit":       ["zinc", "vitamins"],
    "revital":        ["vitamins", "minerals"],

    # Mental health
    "nexito":         ["escitalopram"],
    "stalopam":       ["escitalopram"],
    "clonaz":         ["clonazepam"],
    "rivotril":       ["clonazepam"],
    "alprazolam":     ["alprazolam"],

    # Skin
    "betnovate":      ["betamethasone"],
    "fourderm":       ["clotrimazole", "beclomethasone"],
    "candid":         ["clotrimazole"],

    # Respiratory
    "asthalin":       ["salbutamol", "albuterol"],
    "budecort":       ["budesonide"],
    "foracort":       ["formoterol", "budesonide"],
    "deriphyllin":    ["etofylline", "theophylline"],

    # Common US brands
    "tylenol":        ["acetaminophen", "paracetamol"],
    "advil":          ["ibuprofen"],
    "motrin":         ["ibuprofen"],
    "aleve":          ["naproxen"],
    "benadryl":       ["diphenhydramine"],
    "zyrtec":         ["cetirizine"],
    "claritin":       ["loratadine"],
    "prilosec":       ["omeprazole"],
    "nexium":         ["esomeprazole"],
    "zoloft":         ["sertraline"],
    "prozac":         ["fluoxetine"],
    "xanax":          ["alprazolam"],
    "ambien":         ["zolpidem"],
    "lipitor":        ["atorvastatin"],
    "metoprolol":     ["metoprolol"],
    "lisinopril":     ["lisinopril"],
}


def resolve_brand_to_generic(drug_name: str) -> list[str]:
    """
    Resolves a brand name to generic ingredient names.
    Returns list of generic names, or [drug_name] if not found.
    """
    name_lower = drug_name.lower().strip()

    # ── Step 1: Check local Indian brands dict ─────────────────
    if name_lower in INDIAN_BRANDS:
        result = INDIAN_BRANDS[name_lower]
        print(f"  [local dict] '{drug_name}' → {result}")
        return result

    # ── Step 2: Partial match (e.g. "Montair LC 5" → "montair lc") ──
    for brand, generics in INDIAN_BRANDS.items():
        if name_lower.startswith(brand) or brand.startswith(name_lower):
            print(f"  [partial match] '{drug_name}' → {generics}")
            return generics

    # ── Step 3: RxNorm API ─────────────────────────────────────
    try:
        r = requests.get(
            f"{RXNORM_BASE}/drugs.json",
            params  = {"name": drug_name},
            timeout = 10
        )
        data         = r.json()
        concept_group = data.get("drugGroup", {}).get("conceptGroup", [])
        rxcuis        = []

        for group in concept_group:
            for concept in group.get("conceptProperties", []):
                rxcuis.append(concept["rxcui"])

        if rxcuis:
            rxcui = rxcuis[0]
            r2    = requests.get(
                f"{RXNORM_BASE}/rxcui/{rxcui}/related.json",
                params  = {"tty": "IN"},
                timeout = 10
            )
            data2   = r2.json()
            related = (data2.get("relatedGroup", {})
                           .get("conceptGroup", []))
            generics = []
            for group in related:
                for concept in group.get("conceptProperties", []):
                    name = concept.get("name", "").lower()
                    if name:
                        generics.append(name)
            if generics:
                print(f"  [RxNorm] '{drug_name}' → {generics}")
                return generics

        time.sleep(0.1)

    except Exception as e:
        print(f"  [RxNorm failed] {e}")

    # ── Step 4: Return as-is (already a generic name) ──────────
    print(f"  [no resolution] '{drug_name}' used as-is")
    return [drug_name]


def resolve_all(drug_names: list[str]) -> dict[str, list[str]]:
    result = {}
    for name in drug_names:
        result[name] = resolve_brand_to_generic(name)
    return result


if __name__ == "__main__":
    tests = [
        "Montair LC", "Montair-LC", "Crocin", "Disprin",
        "Combiflam", "Warfarin", "Paracetamol", "Augmentin",
        "Tylenol", "Advil"
    ]
    print("Brand resolution test:")
    for t in tests:
        print(f"  {t:20} → {resolve_brand_to_generic(t)}")