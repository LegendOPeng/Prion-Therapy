#!/bin/bash
echo 'Running 27 simulations...'

echo 'Running opt_vary_G500_G500_N2000_C1000_A2000.mac...'
gears simulations/opt_vary_G500_G500_N2000_C1000_A2000.mac
echo 'Running opt_vary_G1000_G1000_N2000_C1000_A2000.mac...'
gears simulations/opt_vary_G1000_G1000_N2000_C1000_A2000.mac
echo 'Running opt_vary_G2000_G2000_N2000_C1000_A2000.mac...'
gears simulations/opt_vary_G2000_G2000_N2000_C1000_A2000.mac
echo 'Running opt_vary_G5000_G5000_N2000_C1000_A2000.mac...'
gears simulations/opt_vary_G5000_G5000_N2000_C1000_A2000.mac
echo 'Running opt_vary_N500_G2000_N500_C1000_A2000.mac...'
gears simulations/opt_vary_N500_G2000_N500_C1000_A2000.mac
echo 'Running opt_vary_N1000_G2000_N1000_C1000_A2000.mac...'
gears simulations/opt_vary_N1000_G2000_N1000_C1000_A2000.mac
echo 'Running opt_vary_N2000_G2000_N2000_C1000_A2000.mac...'
gears simulations/opt_vary_N2000_G2000_N2000_C1000_A2000.mac
echo 'Running opt_vary_N5000_G2000_N5000_C1000_A2000.mac...'
gears simulations/opt_vary_N5000_G2000_N5000_C1000_A2000.mac
echo 'Running opt_vary_C500_G2000_N2000_C500_A2000.mac...'
gears simulations/opt_vary_C500_G2000_N2000_C500_A2000.mac
echo 'Running opt_vary_C1000_G2000_N2000_C1000_A2000.mac...'
gears simulations/opt_vary_C1000_G2000_N2000_C1000_A2000.mac
echo 'Running opt_vary_C2000_G2000_N2000_C2000_A2000.mac...'
gears simulations/opt_vary_C2000_G2000_N2000_C2000_A2000.mac
echo 'Running opt_vary_C5000_G2000_N2000_C5000_A2000.mac...'
gears simulations/opt_vary_C5000_G2000_N2000_C5000_A2000.mac
echo 'Running opt_vary_A500_G2000_N2000_C1000_A500.mac...'
gears simulations/opt_vary_A500_G2000_N2000_C1000_A500.mac
echo 'Running opt_vary_A1000_G2000_N2000_C1000_A1000.mac...'
gears simulations/opt_vary_A1000_G2000_N2000_C1000_A1000.mac
echo 'Running opt_vary_A2000_G2000_N2000_C1000_A2000.mac...'
gears simulations/opt_vary_A2000_G2000_N2000_C1000_A2000.mac
echo 'Running opt_vary_A5000_G2000_N2000_C1000_A5000.mac...'
gears simulations/opt_vary_A5000_G2000_N2000_C1000_A5000.mac
echo 'Running opt_equal_all500.mac...'
gears simulations/opt_equal_all500.mac
echo 'Running opt_equal_all1000.mac...'
gears simulations/opt_equal_all1000.mac
echo 'Running opt_equal_all2000.mac...'
gears simulations/opt_equal_all2000.mac
echo 'Running opt_equal_all5000.mac...'
gears simulations/opt_equal_all5000.mac
echo 'Running opt_current_5_5_1_5.mac...'
gears simulations/opt_current_5_5_1_5.mac
echo 'Running opt_test_5_5_2_5.mac...'
gears simulations/opt_test_5_5_2_5.mac
echo 'Running opt_test_5_5_5_5.mac...'
gears simulations/opt_test_5_5_5_5.mac
echo 'Running opt_test_2_5_1_5.mac...'
gears simulations/opt_test_2_5_1_5.mac
echo 'Running opt_test_5_2_1_5.mac...'
gears simulations/opt_test_5_2_1_5.mac
echo 'Running opt_test_5_5_05_5.mac...'
gears simulations/opt_test_5_5_05_5.mac
echo 'Running opt_test_1_5_05_5.mac...'
gears simulations/opt_test_1_5_05_5.mac

echo 'Done! Run: python3 simulations/compare_optimize.py'

echo ""
echo "=============================================="
echo "  ALL COUNT OPTIMIZATION SIMULATIONS DONE"
echo ""
echo "  NOW RUN:"
echo "  python3 simulations/compare_optimize.py"
echo "=============================================="
