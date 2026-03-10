## Y-90 Radioembolization

- <https://www.uwmedicine.org/sites/default/files/2019-07/IR-Yttrium-90-Radiotherapy.pdf>

### Yttrium-90

- <https://en.wikipedia.org/wiki/Yttrium-90>
- <https://www.researchgate.net/publication/23233171_Beta_emitters_and_radiation_protection#fullTextFileContent>

### Simulation

```sh
# simulate 10000 Y-90 decays in a liver tumor
gears y90.mac
# open the record for analysis
root record.root
```

Example analysis code in ROOT:

```cpp
// draw beta spectrum
t->Draw("k","trk==4 && stp==0")
// draw energy spectrum of X-rays created by e- bremstrahlung
t->Draw("k","trk==5 && pdg==22 && stp==0")
```
rm prion_therapy.mac
cat > prion_therapy.mac << 'EOF'
/control/verbose 1
/geometry/source brain.tg
/run/initialize

# Co-60 gamma source for low-dose prion therapy
/gps/particle gamma
/gps/energy 1.17 MeV
/gps/pos/type Volume
/gps/pos/shape Sphere
/gps/pos/radius 15 mm
/gps/pos/centre 0 0 0 mm

/tracking/verbose 2
/run/beamOn 100

/control/doifInteractive /vis/open OGL
/vis/drawVolume
/vis/scene/add/axes
/vis/scene/add/trajectories smooth
/vis/scene/endOfEventAction accumulate 1000

/vis/filtering/trajectories/create/particleFilter
/vis/filtering/trajectories/particleFilter-0/add gamma

/run/beamOn 1000
EOF
