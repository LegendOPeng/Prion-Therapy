#!/bin/bash
echo ""
echo "=============================================="
echo "  STAGE 1 — Individual Ray Simulations"
echo "=============================================="

echo ""
echo "Running Gamma (1/4)..."
gears tests/test_gamma.mac
echo "  ✓ Gamma done → gamma_edep.csv"

echo ""
echo "Running Neutron (2/4)..."
gears tests/test_neutron.mac
echo "  ✓ Neutron done → neutron_edep.csv"

echo ""
echo "Running Carbon Ion (3/4)..."
gears tests/test_carbon.mac
echo "  ✓ Carbon done → carbon_edep.csv"

echo ""
echo "Running Alpha (4/4)..."
gears tests/test_alpha.mac
echo "  ✓ Alpha done → alpha_edep.csv"

echo ""
echo "=============================================="
echo "  ALL 4 RAYS DONE"
echo "  CSVs saved: gamma/neutron/carbon/alpha_edep.csv"
echo ""
echo "  NOW RUN:"
echo "  python3 simulations/all_combinations.py"
echo "=============================================="
