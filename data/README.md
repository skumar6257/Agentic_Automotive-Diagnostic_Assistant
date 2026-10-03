# Data Source Files 

This directory contains the raw and processed data for the Vector and Graph databases.

## Sources
1. **DTC Codes (Graph DB):**
   - Source: Open-source OBD-II DTC databases (e.g., GitHub `mytrile/obd-trouble-codes`).
   - Contains: Code, Description, and affected subsystems.
2. **Technical Service Bulletins / Manuals (Vector DB):**
   - Source: [NHTSA Public API](https://api.nhtsa.gov/)
   - Contains: Real manufacturer communications, repair notices, and safety bulletins for specific makes/models (e.g., 2018 Toyota Camry).

Run `python data/fetch_real_data.py` to populate the `raw/` directory.
