#!/bin/bash
echo 'Running 48 simulations...'

echo 'Running seq_GNC_01_GNC.mac...'
gears simulations/seq_GNC_01_GNC.mac
echo 'Running seq_GNC_02_GCN.mac...'
gears simulations/seq_GNC_02_GCN.mac
echo 'Running seq_GNC_03_NGC.mac...'
gears simulations/seq_GNC_03_NGC.mac
echo 'Running seq_GNC_04_NCG.mac...'
gears simulations/seq_GNC_04_NCG.mac
echo 'Running seq_GNC_05_CGN.mac...'
gears simulations/seq_GNC_05_CGN.mac
echo 'Running seq_GNC_06_CNG.mac...'
gears simulations/seq_GNC_06_CNG.mac
echo 'Running seq_GNA_01_GNA.mac...'
gears simulations/seq_GNA_01_GNA.mac
echo 'Running seq_GNA_02_GAN.mac...'
gears simulations/seq_GNA_02_GAN.mac
echo 'Running seq_GNA_03_NGA.mac...'
gears simulations/seq_GNA_03_NGA.mac
echo 'Running seq_GNA_04_NAG.mac...'
gears simulations/seq_GNA_04_NAG.mac
echo 'Running seq_GNA_05_AGN.mac...'
gears simulations/seq_GNA_05_AGN.mac
echo 'Running seq_GNA_06_ANG.mac...'
gears simulations/seq_GNA_06_ANG.mac
echo 'Running seq_GCA_01_GCA.mac...'
gears simulations/seq_GCA_01_GCA.mac
echo 'Running seq_GCA_02_GAC.mac...'
gears simulations/seq_GCA_02_GAC.mac
echo 'Running seq_GCA_03_CGA.mac...'
gears simulations/seq_GCA_03_CGA.mac
echo 'Running seq_GCA_04_CAG.mac...'
gears simulations/seq_GCA_04_CAG.mac
echo 'Running seq_GCA_05_AGC.mac...'
gears simulations/seq_GCA_05_AGC.mac
echo 'Running seq_GCA_06_ACG.mac...'
gears simulations/seq_GCA_06_ACG.mac
echo 'Running seq_NCA_01_NCA.mac...'
gears simulations/seq_NCA_01_NCA.mac
echo 'Running seq_NCA_02_NAC.mac...'
gears simulations/seq_NCA_02_NAC.mac
echo 'Running seq_NCA_03_CNA.mac...'
gears simulations/seq_NCA_03_CNA.mac
echo 'Running seq_NCA_04_CAN.mac...'
gears simulations/seq_NCA_04_CAN.mac
echo 'Running seq_NCA_05_ANC.mac...'
gears simulations/seq_NCA_05_ANC.mac
echo 'Running seq_NCA_06_ACN.mac...'
gears simulations/seq_NCA_06_ACN.mac
echo 'Running seq_GNCA_01_GNCA.mac...'
gears simulations/seq_GNCA_01_GNCA.mac
echo 'Running seq_GNCA_02_GNAC.mac...'
gears simulations/seq_GNCA_02_GNAC.mac
echo 'Running seq_GNCA_03_GCNA.mac...'
gears simulations/seq_GNCA_03_GCNA.mac
echo 'Running seq_GNCA_04_GCAN.mac...'
gears simulations/seq_GNCA_04_GCAN.mac
echo 'Running seq_GNCA_05_GANC.mac...'
gears simulations/seq_GNCA_05_GANC.mac
echo 'Running seq_GNCA_06_GACN.mac...'
gears simulations/seq_GNCA_06_GACN.mac
echo 'Running seq_GNCA_07_NGCA.mac...'
gears simulations/seq_GNCA_07_NGCA.mac
echo 'Running seq_GNCA_08_NGAC.mac...'
gears simulations/seq_GNCA_08_NGAC.mac
echo 'Running seq_GNCA_09_NCGA.mac...'
gears simulations/seq_GNCA_09_NCGA.mac
echo 'Running seq_GNCA_10_NCAG.mac...'
gears simulations/seq_GNCA_10_NCAG.mac
echo 'Running seq_GNCA_11_NAGC.mac...'
gears simulations/seq_GNCA_11_NAGC.mac
echo 'Running seq_GNCA_12_NACG.mac...'
gears simulations/seq_GNCA_12_NACG.mac
echo 'Running seq_GNCA_13_CGNA.mac...'
gears simulations/seq_GNCA_13_CGNA.mac
echo 'Running seq_GNCA_14_CGAN.mac...'
gears simulations/seq_GNCA_14_CGAN.mac
echo 'Running seq_GNCA_15_CNGA.mac...'
gears simulations/seq_GNCA_15_CNGA.mac
echo 'Running seq_GNCA_16_CNAG.mac...'
gears simulations/seq_GNCA_16_CNAG.mac
echo 'Running seq_GNCA_17_CAGN.mac...'
gears simulations/seq_GNCA_17_CAGN.mac
echo 'Running seq_GNCA_18_CANG.mac...'
gears simulations/seq_GNCA_18_CANG.mac
echo 'Running seq_GNCA_19_AGNC.mac...'
gears simulations/seq_GNCA_19_AGNC.mac
echo 'Running seq_GNCA_20_AGCN.mac...'
gears simulations/seq_GNCA_20_AGCN.mac
echo 'Running seq_GNCA_21_ANGC.mac...'
gears simulations/seq_GNCA_21_ANGC.mac
echo 'Running seq_GNCA_22_ANCG.mac...'
gears simulations/seq_GNCA_22_ANCG.mac
echo 'Running seq_GNCA_23_ACGN.mac...'
gears simulations/seq_GNCA_23_ACGN.mac
echo 'Running seq_GNCA_24_ACNG.mac...'
gears simulations/seq_GNCA_24_ACNG.mac

echo 'All done! Now run: python3 compare_orders.py'

echo ""
echo "=============================================="
echo "  ALL 48 FIRING ORDER SIMULATIONS DONE"
echo ""
echo "  NOW RUN:"
echo "  python3 simulations/compare_orders.py"
echo "=============================================="
