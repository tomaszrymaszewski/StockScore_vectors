"""
Build the GICS hierarchy map: sub_industry -> industry -> industry_group -> sector.

The iShares Russell 3000 holdings file only carries a GICS *sub-industry* label per
ticker. To evaluate the embedding's hit-rate at coarser GICS granularities (the
"granularity ladder" in the paper) we need to roll each sub-industry up to its parent
industry, industry group, and sector.

This script encodes the GICS 2023 structure (effective March 2023, the vintage the
holdings file uses -- e.g. "Broadline Retail", "Passenger Airlines",
"Consumer Staples Merchandise Retail") for exactly the sub-industries that appear in the
holdings file, and writes them to data/gics_hierarchy.csv.

A completeness check asserts that every (non-junk) sub-industry present in the holdings
file is covered by the map, so an incomplete map fails loudly rather than silently
dropping stocks from the coarse-granularity evaluation.

Run:  python -m jobs.build_gics_hierarchy
"""
from __future__ import annotations

import os
import sys

import pandas as pd

HOLDINGS_FILE = os.path.join("data", "Holdings_details_Russell_3000_ETF.csv")
OUTPUT_FILE = os.path.join("data", "gics_hierarchy.csv")

# Parsing artifacts / footer rows that leak into the SUB-INDUSTRY column. Not real GICS.
JUNK_SUBINDUSTRIES = {"---", "MARKET VALUE*", "SHARES", "", "nan"}


# sector -> { industry_group -> { industry -> [sub_industries] } }
HIERARCHY: dict[str, dict[str, dict[str, list[str]]]] = {
    "Energy": {
        "Energy": {
            "Energy Equipment & Services": [
                "Oil & Gas Drilling",
                "Oil & Gas Equipment & Services",
            ],
            "Oil, Gas & Consumable Fuels": [
                "Integrated Oil & Gas",
                "Oil & Gas Exploration & Production",
                "Oil & Gas Refining & Marketing",
                "Oil & Gas Storage & Transportation",
                "Coal & Consumable Fuels",
            ],
        },
    },
    "Materials": {
        "Materials": {
            "Chemicals": [
                "Commodity Chemicals",
                "Diversified Chemicals",
                "Fertilizers & Agricultural Chemicals",
                "Industrial Gases",
                "Specialty Chemicals",
            ],
            "Construction Materials": ["Construction Materials"],
            "Containers & Packaging": [
                "Metal, Glass & Plastic Containers",
                "Paper & Plastic Packaging Products & Materials",
            ],
            "Metals & Mining": [
                "Aluminum",
                "Copper",
                "Diversified Metals & Mining",
                "Gold",
                "Precious Metals & Minerals",
                "Silver",
                "Steel",
            ],
            "Paper & Forest Products": [
                "Forest Products",
                "Paper Products",
            ],
        },
    },
    "Industrials": {
        "Capital Goods": {
            "Aerospace & Defense": ["Aerospace & Defense"],
            "Building Products": ["Building Products"],
            "Construction & Engineering": ["Construction & Engineering"],
            "Electrical Equipment": [
                "Electrical Components & Equipment",
                "Heavy Electrical Equipment",
            ],
            "Industrial Conglomerates": ["Industrial Conglomerates"],
            "Machinery": [
                "Construction Machinery & Heavy Transportation Equipment",
                "Agricultural & Farm Machinery",
                "Industrial Machinery & Supplies & Components",
            ],
            "Trading Companies & Distributors": ["Trading Companies & Distributors"],
        },
        "Commercial & Professional Services": {
            "Commercial Services & Supplies": [
                "Commercial Printing",
                "Environmental & Facilities Services",
                "Office Services & Supplies",
                "Diversified Support Services",
                "Security & Alarm Services",
            ],
            "Professional Services": [
                "Human Resource & Employment Services",
                "Research & Consulting Services",
            ],
        },
        "Transportation": {
            "Air Freight & Logistics": ["Air Freight & Logistics"],
            "Passenger Airlines": ["Passenger Airlines"],
            "Marine Transportation": ["Marine Transportation"],
            "Ground Transportation": [
                "Rail Transportation",
                "Cargo Ground Transportation",
                "Passenger Ground Transportation",
            ],
            "Transportation Infrastructure": ["Airport Services"],
        },
    },
    "Consumer Discretionary": {
        "Automobiles & Components": {
            "Automobile Components": [
                "Automotive Parts & Equipment",
                "Tires & Rubber",
            ],
            "Automobiles": [
                "Automobile Manufacturers",
                "Motorcycle Manufacturers",
            ],
        },
        "Consumer Durables & Apparel": {
            "Household Durables": [
                "Consumer Electronics",
                "Home Furnishings",
                "Homebuilding",
                "Household Appliances",
                "Housewares & Specialties",
            ],
            "Leisure Products": ["Leisure Products"],
            "Textiles, Apparel & Luxury Goods": [
                "Apparel, Accessories & Luxury Goods",
                "Footwear",
            ],
        },
        "Consumer Services": {
            "Hotels, Restaurants & Leisure": [
                "Casinos & Gaming",
                "Hotels, Resorts & Cruise Lines",
                "Leisure Facilities",
                "Restaurants",
            ],
            "Diversified Consumer Services": [
                "Education Services",
                "Specialized Consumer Services",
            ],
        },
        "Consumer Discretionary Distribution & Retail": {
            "Distributors": ["Distributors"],
            "Broadline Retail": ["Broadline Retail"],
            "Specialty Retail": [
                "Apparel Retail",
                "Computer & Electronics Retail",
                "Home Improvement Retail",
                "Homefurnishing Retail",
                "Automotive Retail",
                "Other Specialty Retail",
            ],
        },
    },
    "Consumer Staples": {
        "Consumer Staples Distribution & Retail": {
            "Consumer Staples Distribution & Retail": [
                "Consumer Staples Merchandise Retail",
                "Food Distributors",
                "Food Retail",
            ],
        },
        "Food, Beverage & Tobacco": {
            "Beverages": [
                "Brewers",
                "Distillers & Vintners",
                "Soft Drinks & Non-alcoholic Beverages",
            ],
            "Food Products": [
                "Agricultural Products & Services",
                "Packaged Foods & Meats",
            ],
            "Tobacco": ["Tobacco"],
        },
        "Household & Personal Products": {
            "Household Products": ["Household Products"],
            "Personal Care Products": ["Personal Care Products"],
        },
    },
    "Health Care": {
        "Health Care Equipment & Services": {
            "Health Care Equipment & Supplies": [
                "Health Care Equipment",
                "Health Care Supplies",
            ],
            "Health Care Providers & Services": [
                "Health Care Distributors",
                "Health Care Services",
                "Health Care Facilities",
                "Managed Health Care",
            ],
            "Health Care Technology": ["Health Care Technology"],
        },
        "Pharmaceuticals, Biotechnology & Life Sciences": {
            "Biotechnology": ["Biotechnology"],
            "Pharmaceuticals": ["Pharmaceuticals"],
            "Life Sciences Tools & Services": ["Life Sciences Tools & Services"],
        },
    },
    "Financials": {
        "Banks": {
            "Banks": [
                "Diversified Banks",
                "Regional Banks",
            ],
        },
        "Financial Services": {
            "Financial Services": [
                "Diversified Financial Services",
                "Multi-Sector Holdings",
                "Specialized Finance",
                "Commercial & Residential Mortgage Finance",
                "Transaction & Payment Processing Services",
            ],
            "Consumer Finance": ["Consumer Finance"],
            "Capital Markets": [
                "Asset Management & Custody Banks",
                "Investment Banking & Brokerage",
                "Diversified Capital Markets",
                "Financial Exchanges & Data",
            ],
            "Mortgage Real Estate Investment Trusts (REITs)": ["Mortgage REITs"],
        },
        "Insurance": {
            "Insurance": [
                "Insurance Brokers",
                "Life & Health Insurance",
                "Multi-line Insurance",
                "Property & Casualty Insurance",
                "Reinsurance",
            ],
        },
    },
    "Information Technology": {
        "Software & Services": {
            "IT Services": [
                "IT Consulting & Other Services",
                "Internet Services & Infrastructure",
                "Data Processing & Outsourced Services",
            ],
            "Software": [
                "Application Software",
                "Systems Software",
            ],
        },
        "Technology Hardware & Equipment": {
            "Communications Equipment": ["Communications Equipment"],
            "Technology Hardware, Storage & Peripherals": [
                "Technology Hardware, Storage & Peripherals",
            ],
            "Electronic Equipment, Instruments & Components": [
                "Electronic Components",
                "Electronic Equipment & Instruments",
                "Electronic Manufacturing Services",
                "Technology Distributors",
            ],
        },
        "Semiconductors & Semiconductor Equipment": {
            "Semiconductors & Semiconductor Equipment": [
                "Semiconductors",
                "Semiconductor Materials & Equipment",
            ],
        },
    },
    "Communication Services": {
        "Telecommunication Services": {
            "Diversified Telecommunication Services": [
                "Integrated Telecommunication Services",
                "Alternative Carriers",
            ],
            "Wireless Telecommunication Services": ["Wireless Telecommunication Services"],
        },
        "Media & Entertainment": {
            "Media": [
                "Advertising",
                "Broadcasting",
                "Cable & Satellite",
                "Publishing",
            ],
            "Entertainment": [
                "Movies & Entertainment",
                "Interactive Home Entertainment",
            ],
            "Interactive Media & Services": ["Interactive Media & Services"],
        },
    },
    "Utilities": {
        "Utilities": {
            "Electric Utilities": ["Electric Utilities"],
            "Gas Utilities": ["Gas Utilities"],
            "Multi-Utilities": ["Multi-Utilities"],
            "Water Utilities": ["Water Utilities"],
            "Independent Power and Renewable Electricity Producers": [
                "Independent Power Producers & Energy Traders",
                "Renewable Electricity",
            ],
        },
    },
    "Real Estate": {
        "Equity Real Estate Investment Trusts (REITs)": {
            "Diversified REITs": ["Diversified REITs"],
            "Industrial REITs": ["Industrial REITs"],
            "Hotel & Resort REITs": ["Hotel & Resort REITs"],
            "Office REITs": ["Office REITs"],
            "Health Care REITs": ["Health Care REITs"],
            "Residential REITs": [
                "Multi-Family Residential REITs",
                "Single-Family Residential REITs",
            ],
            "Retail REITs": ["Retail REITs"],
            "Specialized REITs": [
                "Data Center REITs",
                "Self-Storage REITs",
                "Telecom Tower REITs",
                "Timber REITs",
                "Other Specialized REITs",
            ],
        },
        "Real Estate Management & Development": {
            "Real Estate Management & Development": [
                "Diversified Real Estate Activities",
                "Real Estate Development",
                "Real Estate Operating Companies",
                "Real Estate Services",
            ],
        },
    },
}


def build_rows() -> list[dict]:
    rows = []
    for sector, groups in HIERARCHY.items():
        for industry_group, industries in groups.items():
            for industry, subs in industries.items():
                for sub in subs:
                    rows.append(
                        {
                            "sub_industry": sub,
                            "industry": industry,
                            "industry_group": industry_group,
                            "sector": sector,
                        }
                    )
    return rows


def load_holdings_subindustries() -> set[str]:
    """Read the distinct, non-junk sub-industry strings from the holdings file."""
    header_idx = None
    with open(HOLDINGS_FILE) as f:
        for i, line in enumerate(f):
            if "TICKER" in line and "SUB-INDUSTRY" in line:
                header_idx = i
                break
    if header_idx is None:
        raise RuntimeError(f"Could not find header row in {HOLDINGS_FILE}")

    df = pd.read_csv(HOLDINGS_FILE, skiprows=header_idx)
    df.columns = [c.strip() for c in df.columns]
    vals = df["SUB-INDUSTRY"].dropna().astype(str).str.strip()
    distinct = set(vals.unique())
    # Drop junk + anything that looks like a currency/number artifact.
    return {
        s
        for s in distinct
        if s not in JUNK_SUBINDUSTRIES and not s.startswith(("$", "-$"))
    }


def main() -> int:
    rows = build_rows()
    mapped = {r["sub_industry"] for r in rows}

    # Sanity: no duplicate sub-industry across two parents.
    if len(mapped) != len(rows):
        from collections import Counter

        dupes = [s for s, c in Counter(r["sub_industry"] for r in rows).items() if c > 1]
        raise RuntimeError(f"Duplicate sub-industries in HIERARCHY: {dupes}")

    holdings_subs = load_holdings_subindustries()
    missing = sorted(holdings_subs - mapped)
    if missing:
        print("❌ Sub-industries present in holdings but missing from HIERARCHY:")
        for m in missing:
            print(f"   - {m}")
        return 1

    extra = sorted(mapped - holdings_subs)
    if extra:
        # Not fatal: the map may cover sub-industries not in this particular file.
        print(f"ℹ️  {len(extra)} mapped sub-industries are not in the current holdings file.")

    out = pd.DataFrame(rows).sort_values(
        ["sector", "industry_group", "industry", "sub_industry"]
    )
    out.to_csv(OUTPUT_FILE, index=False)

    n_sectors = out["sector"].nunique()
    n_groups = out["industry_group"].nunique()
    n_inds = out["industry"].nunique()
    n_subs = out["sub_industry"].nunique()
    print(f"✅ Wrote {OUTPUT_FILE}")
    print(
        f"   {n_subs} sub-industries -> {n_inds} industries -> "
        f"{n_groups} industry groups -> {n_sectors} sectors"
    )
    print(f"   Covered all {len(holdings_subs)} sub-industries in the holdings file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
