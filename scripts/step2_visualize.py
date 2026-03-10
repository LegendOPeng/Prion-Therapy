import json, os, math, subprocess, sys

# ── Check matplotlib available ────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')  # no display needed — saves to file
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("Installing matplotlib...")
    subprocess.run([sys.executable,"-m","pip","install","matplotlib","--break-system-packages"],
                   capture_output=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec

LOG_FILE   = "simulation_progress_log.json"
PLOT_DIR   = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Colors per ray — distinct and colorblind-friendly
RAY_COLORS = {
    "Gamma":      "#E63946",   # red
    "Neutron":    "#2196F3",   # blue
    "Carbon Ion": "#FF9800",   # orange
    "Alpha":      "#4CAF50",   # green
}
ANCHOR_COLOR = "#555555"
ALGO_COLOR   = "#9C27B0"   # purple for algo-picked points
OPTIMAL_COLOR= "#FFD700"   # gold for optimal

DOSE_LIMITS = {"Gamma":24.0,"Neutron":14.0,"Carbon Ion":20.0,"Alpha":20.0}
MEV_TO_JOULE=1.60218e-13
CORE_MASS_KG=1.0e-6*1350

def dose_gy(core_mev): return core_mev*MEV_TO_JOULE/CORE_MASS_KG

def fit_quad(xs,ys):
    n=len(xs)
    if n<3: return None
    s1=sum(xs); s2=sum(x**2 for x in xs); s3=sum(x**3 for x in xs); s4=sum(x**4 for x in xs)
    t0=sum(ys); t1=sum(x*y for x,y in zip(xs,ys)); t2=sum(x**2*y for x,y in zip(xs,ys))
    A=[[s4,s3,s2],[s3,s2,s1],[s2,s1,float(n)]]
    b=[t2,t1,t0]
    for col in range(3):
        piv=col+max(range(3-col),key=lambda r:abs(A[col+r][col]))
        A[col],A[piv]=A[piv],A[col]; b[col],b[piv]=b[piv],b[col]
        if abs(A[col][col])<1e-20: return None
        for row in range(col+1,3):
            f=A[row][col]/A[col][col]
            for k in range(col,3): A[row][k]-=f*A[col][k]
            b[row]-=f*b[col]
    x=[0.0]*3
    for i in range(2,-1,-1):
        x[i]=b[i]
        for j in range(i+1,3): x[i]-=A[i][j]*x[j]
        x[i]/=A[i][i]
    return x[0],x[1],x[2]

def quad_curve(a,b,c,x_lo,x_hi,n=300):
    xs=[x_lo+(x_hi-x_lo)*i/(n-1) for i in range(n)]
    ys=[a*x**2+b*x+c for x in xs]
    return xs,ys

# ── LOAD LOG ──────────────────────────────────────────────────────────────────
if not os.path.exists(LOG_FILE):
    print(f"ERROR: {LOG_FILE} not found. Run Step 2 first."); exit(1)
with open(LOG_FILE) as f: log=json.load(f)
if "step2" not in log or "search" not in log["step2"]:
    print("ERROR: Step 2 results not in log. Run Step 2 first."); exit(1)

search=log["step2"]["search"]
optimal=log["step2"]["optimal"]
rays=list(search.keys())
print(f"Loaded Step 2 data for: {rays}")

plot_paths={}

# ── PER-RAY PLOTS ─────────────────────────────────────────────────────────────
for ray in rays:
    color=RAY_COLORS.get(ray,"#333333")
    rdata=search[ray]
    tested=rdata["tested"]       # str keys
    anchors=set(rdata.get("anchors",[]))
    best_count=rdata["optimal_count"]
    history=rdata.get("search_history",[])

    counts_int=sorted(int(c) for c in tested.keys())
    effs=[tested[str(c)]["efficiency"] for c in counts_int]
    cvs=[tested[str(c)]["final_cv_pct"] for c in counts_int]
    doses=[dose_gy(tested[str(c)]["core_MeV"]["mean"]) for c in counts_int]
    core_mevs=[tested[str(c)]["core_MeV"]["mean"] for c in counts_int]
    core_stds=[tested[str(c)]["core_MeV"]["std"] for c in counts_int]

    fig=plt.figure(figsize=(14,10))
    fig.patch.set_facecolor('#0F0F1A')
    gs=GridSpec(2,2,figure=fig,hspace=0.42,wspace=0.35)

    # ── Plot 1: Efficiency curve + quadratic fit ───────────────────────────
    ax1=fig.add_subplot(gs[0,:])   # full width top
    ax1.set_facecolor('#16162A')

    # Fit quadratic to all points for smooth curve
    result=fit_quad([float(c) for c in counts_int],effs)
    if result:
        a,b,c_=result
        lo,hi=min(counts_int),max(counts_int)
        margin=(hi-lo)*0.05
        cx,cy=quad_curve(a,b,c_,lo-margin,hi+margin)
        ax1.plot(cx,cy,color=color,alpha=0.4,linewidth=2,linestyle='--',
                 label='Quadratic fit')
        if a<0:
            peak=-b/(2*a)
            if lo<=peak<=hi:
                peak_eff=a*peak**2+b*peak+c_
                ax1.axvline(peak,color=OPTIMAL_COLOR,linewidth=1.5,
                           linestyle=':',alpha=0.7)
                ax1.annotate(f'Predicted\npeak\n{int(round(peak))} particles',
                            xy=(peak,peak_eff),
                            xytext=(peak+(hi-lo)*0.08, peak_eff*0.97),
                            color=OPTIMAL_COLOR,fontsize=8,
                            arrowprops=dict(arrowstyle='->',color=OPTIMAL_COLOR,lw=1))

    # Plot points — color by source (anchor vs algo-pick)
    for c,e in zip(counts_int,effs):
        is_anchor=c in anchors
        is_best=c==best_count
        marker='*' if is_best else ('s' if is_anchor else 'o')
        msize=180 if is_best else (80 if is_anchor else 60)
        ec=OPTIMAL_COLOR if is_best else (ANCHOR_COLOR if is_anchor else ALGO_COLOR)
        fc=OPTIMAL_COLOR if is_best else (color if is_anchor else ALGO_COLOR)
        ax1.scatter(c,e,s=msize,c=fc,edgecolors=ec,linewidths=1.5,
                   marker=marker,zorder=5)

    # Connect with line
    ax1.plot(counts_int,effs,color=color,linewidth=1.2,alpha=0.6)

    # Mark optimal
    best_eff=tested[str(best_count)]["efficiency"]
    ax1.annotate(f'OPTIMAL\n{best_count} particles\neff={best_eff:.4f}',
                xy=(best_count,best_eff),
                xytext=(best_count,best_eff*1.08 if best_eff>0 else best_eff-0.01),
                ha='center',color=OPTIMAL_COLOR,fontsize=9,fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3',facecolor='#1A1A2E',
                         edgecolor=OPTIMAL_COLOR,alpha=0.9))

    ax1.set_xlabel('Particle Count (beamOn)',color='#CCCCCC',fontsize=10)
    ax1.set_ylabel('Efficiency (core MeV / 1000 particles)',color='#CCCCCC',fontsize=10)
    ax1.set_title(f'{ray} — Efficiency Curve & Interpolation Search',
                  color='white',fontsize=13,fontweight='bold',pad=12)
    ax1.tick_params(colors='#AAAAAA')
    for sp in ax1.spines.values(): sp.set_color('#333355')
    ax1.grid(True,color='#222244',linewidth=0.5,alpha=0.7)

    # Legend
    handles=[
        mpatches.Patch(facecolor=color,edgecolor=ANCHOR_COLOR,label='Anchor point (log-spaced start)'),
        mpatches.Patch(facecolor=ALGO_COLOR,label='Algorithm-picked point'),
        mpatches.Patch(facecolor=OPTIMAL_COLOR,label='Optimal count (highest eff)'),
        plt.Line2D([0],[0],color=color,linestyle='--',alpha=0.4,label='Quadratic fit'),
    ]
    ax1.legend(handles=handles,loc='upper right',framealpha=0.2,
               labelcolor='#CCCCCC',fontsize=8)

    # ── Plot 2: Core MeV vs count (with error bars) ───────────────────────
    ax2=fig.add_subplot(gs[1,0])
    ax2.set_facecolor('#16162A')
    ax2.errorbar(counts_int,core_mevs,yerr=core_stds,
                fmt='o-',color=color,ecolor='#888888',
                elinewidth=1,capsize=4,linewidth=1.5,markersize=5)
    ax2.axvline(best_count,color=OPTIMAL_COLOR,linewidth=1.5,
               linestyle=':',alpha=0.8,label=f'Optimal ({best_count})')
    ax2.set_xlabel('Particle Count',color='#CCCCCC',fontsize=9)
    ax2.set_ylabel('Core MeV deposited (mean ± std)',color='#CCCCCC',fontsize=9)
    ax2.set_title('Core Energy Deposition',color='white',fontsize=10,fontweight='bold')
    ax2.tick_params(colors='#AAAAAA'); ax2.grid(True,color='#222244',linewidth=0.5)
    for sp in ax2.spines.values(): sp.set_color('#333355')
    ax2.legend(labelcolor='#CCCCCC',fontsize=8,framealpha=0.2)

    # ── Plot 3: Dose estimate vs count + safety line ──────────────────────
    ax3=fig.add_subplot(gs[1,1])
    ax3.set_facecolor('#16162A')
    safe_doses=[d for d in doses if d<=DOSE_LIMITS[ray]]
    safe_counts=[c for c,d in zip(counts_int,doses) if d<=DOSE_LIMITS[ray]]
    unsafe_doses=[d for d in doses if d>DOSE_LIMITS[ray]]
    unsafe_counts=[c for c,d in zip(counts_int,doses) if d>DOSE_LIMITS[ray]]

    if safe_counts:
        ax3.scatter(safe_counts,safe_doses,color='#4CAF50',s=50,
                   label='Safe',zorder=5)
    if unsafe_counts:
        ax3.scatter(unsafe_counts,unsafe_doses,color='#E63946',s=50,
                   marker='X',label='Unsafe',zorder=5)
    ax3.plot(counts_int,doses,color=color,linewidth=1.2,alpha=0.5)
    ax3.axhline(DOSE_LIMITS[ray],color='#FF5722',linewidth=2,linestyle='--',
               label=f'Brain limit ({DOSE_LIMITS[ray]} Gy)')
    ax3.axvline(best_count,color=OPTIMAL_COLOR,linewidth=1.5,
               linestyle=':',alpha=0.8)
    ax3.set_xlabel('Particle Count',color='#CCCCCC',fontsize=9)
    ax3.set_ylabel('Approx Dose (Gy)',color='#CCCCCC',fontsize=9)
    ax3.set_title('Dose Estimate vs Safety Limit',color='white',
                  fontsize=10,fontweight='bold')
    ax3.tick_params(colors='#AAAAAA'); ax3.grid(True,color='#222244',linewidth=0.5)
    for sp in ax3.spines.values(): sp.set_color('#333355')
    ax3.legend(labelcolor='#CCCCCC',fontsize=8,framealpha=0.2)

    # ── Search path annotation (bottom of fig) ────────────────────────────
    if history:
        path_str="Search path: " + " → ".join(
            [f"{h['snapped']}" for h in history[:8]])
        if len(history)>8: path_str+=" → ..."
        path_str+=f"  [converged at {best_count}]"
        fig.text(0.5,0.01,path_str,ha='center',va='bottom',
                color='#888888',fontsize=8,style='italic')

    plt.suptitle(
        f'{ray} — Step 2 Interpolation Search Results\n'
        f'Optimal: {best_count} particles  |  '
        f'Core MeV: {core_mevs[counts_int.index(best_count)] if best_count in counts_int else "—":,.1f}  |  '
        f'Eff: {best_eff:.5f} MeV/1kP  |  '
        f'{"SAFE ✓" if optimal[ray]["safety_ok"] else "UNSAFE ✗"}',
        color='white',fontsize=11,y=0.98,
        bbox=dict(boxstyle='round',facecolor='#1A1A2E',alpha=0.8,
                 edgecolor=color,linewidth=2)
    )

    path=os.path.join(PLOT_DIR,f"step2_{ray.replace(' ','_').lower()}_curve.png")
    plt.savefig(path,dpi=150,bbox_inches='tight',facecolor='#0F0F1A')
    plt.close()
    plot_paths[ray]=path
    print(f"  Saved: {path}")

# ── COMBINED OVERVIEW PLOT ────────────────────────────────────────────────────
print("\nGenerating combined overview...")
fig,axes=plt.subplots(2,2,figsize=(16,10))
fig.patch.set_facecolor('#0F0F1A')

for idx,(ray,ax) in enumerate(zip(rays,axes.flat)):
    color=RAY_COLORS.get(ray,"#333333")
    rdata=search[ray]
    tested_r=rdata["tested"]
    anchors_r=set(rdata.get("anchors",[]))
    best_c=rdata["optimal_count"]

    counts_i=sorted(int(c) for c in tested_r.keys())
    effs_i=[tested_r[str(c)]["efficiency"] for c in counts_i]

    ax.set_facecolor('#16162A')
    result=fit_quad([float(c) for c in counts_i],effs_i)
    if result:
        a,b,c_=result
        lo,hi=min(counts_i),max(counts_i)
        margin=(hi-lo)*0.05
        cx,cy=quad_curve(a,b,c_,lo-margin,hi+margin)
        ax.plot(cx,cy,color=color,alpha=0.35,linewidth=2,linestyle='--')

    for c,e in zip(counts_i,effs_i):
        is_best=c==best_c
        fc=OPTIMAL_COLOR if is_best else (color if c in anchors_r else ALGO_COLOR)
        mk='*' if is_best else 'o'
        ms=150 if is_best else 50
        ax.scatter(c,e,s=ms,c=fc,zorder=5,marker=mk)
    ax.plot(counts_i,effs_i,color=color,linewidth=1,alpha=0.5)
    ax.axvline(best_c,color=OPTIMAL_COLOR,linewidth=1.5,linestyle=':',alpha=0.7)

    best_e=tested_r[str(best_c)]["efficiency"]
    safe=optimal[ray]["safety_ok"]
    ax.set_title(f'{ray}  |  Optimal: {best_c}p  |  {"SAFE ✓" if safe else "UNSAFE ✗"}',
                color='white',fontsize=10,fontweight='bold')
    ax.set_xlabel('Particle Count',color='#CCCCCC',fontsize=8)
    ax.set_ylabel('Efficiency (MeV/1kP)',color='#CCCCCC',fontsize=8)
    ax.tick_params(colors='#AAAAAA',labelsize=7)
    ax.grid(True,color='#222244',linewidth=0.5,alpha=0.6)
    for sp in ax.spines.values(): sp.set_color('#333355')

    ax.annotate(f'eff={best_e:.4f}',xy=(best_c,best_e),
               xytext=(best_c,best_e*1.1 if best_e>0 else best_e-0.005),
               ha='center',color=OPTIMAL_COLOR,fontsize=8,fontweight='bold')

plt.suptitle('Step 2 — Interpolation Search: All Rays Overview',
            color='white',fontsize=14,fontweight='bold',y=1.01)
plt.tight_layout()
overview_path=os.path.join(PLOT_DIR,"step2_all_rays_overview.png")
plt.savefig(overview_path,dpi=150,bbox_inches='tight',facecolor='#0F0F1A')
plt.close()
plot_paths["_overview"]=overview_path
print(f"  Saved: {overview_path}")

# ── BUILD WORD DOC ────────────────────────────────────────────────────────────
print("\nBuilding Word document...")
try:
    import subprocess as sp
    result=sp.run(["node","--version"],capture_output=True)
    has_node=result.returncode==0
except: has_node=False

if has_node:
    # Write JS to build docx with embedded images
    js=r"""
const {Document,Packer,Paragraph,TextRun,ImageRun,AlignmentType,
       HeadingLevel,BorderStyle,WidthType,ShadingType,Header,
       PageNumberElement,Footer,TableRow,TableCell,Table} = require('docx');
const fs=require('fs');
const path=require('path');

const TEAL="1A6B72",GOLD="B8860B",BLACK="111111",GRAY="F5F5F5";
const b={style:BorderStyle.SINGLE,size:1,color:"CCCCCC"};
const borders={top:b,bottom:b,left:b,right:b};

const p=(text,opts={})=>new Paragraph({
  alignment:opts.align||AlignmentType.LEFT,
  spacing:{before:opts.before||0,after:opts.after||100},
  children:[new TextRun({text,font:"Arial",size:opts.size||22,
    bold:opts.bold||false,italic:opts.italic||false,color:opts.color||BLACK})]
});
const h1=t=>new Paragraph({heading:HeadingLevel.HEADING_1,
  spacing:{before:280,after:140},
  border:{bottom:{style:BorderStyle.SINGLE,size:8,color:TEAL,space:2}},
  children:[new TextRun({text:t,font:"Arial",size:32,bold:true,color:TEAL})]});
const h2=t=>new Paragraph({heading:HeadingLevel.HEADING_2,
  spacing:{before:200,after:100},
  children:[new TextRun({text:t,font:"Arial",size:26,bold:true,color:GOLD})]});

function imgPara(imgPath,widthEmu,heightEmu){
  try{
    const buf=fs.readFileSync(imgPath);
    return new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:120,after:120},
      children:[new ImageRun({data:buf,transformation:{width:widthEmu,height:heightEmu},
        type:"png"})]});
  }catch(e){
    return p(`[Image not found: ${imgPath}]`,{italic:true,color:"888888"});
  }
}

function cell(text,opts={}){
  return new TableCell({borders,
    width:{size:opts.w||4680,type:WidthType.DXA},
    shading:opts.shade?{fill:opts.shade,type:ShadingType.CLEAR}:undefined,
    margins:{top:80,bottom:80,left:120,right:120},
    children:[new Paragraph({alignment:opts.align||AlignmentType.LEFT,
      children:[new TextRun({text,font:"Arial",size:opts.size||20,
        bold:opts.bold||false,color:opts.color||BLACK})]})]});
}

// Load log
const log=JSON.parse(fs.readFileSync("simulation_progress_log.json"));
const search=log.step2.search;
const optimal=log.step2.optimal;
const rays=Object.keys(search);
const timing=log.step2.timing||{};

// Summary table rows
const summaryRows=[
  new TableRow({children:[
    cell("Ray",{w:1500,shade:TEAL,bold:true,color:"FFFFFF",size:18}),
    cell("Optimal Count",{w:1500,shade:TEAL,bold:true,color:"FFFFFF",size:18,align:AlignmentType.CENTER}),
    cell("Core MeV",{w:2000,shade:TEAL,bold:true,color:"FFFFFF",size:18,align:AlignmentType.CENTER}),
    cell("Eff MeV/1kP",{w:1800,shade:TEAL,bold:true,color:"FFFFFF",size:18,align:AlignmentType.CENTER}),
    cell("Dose Gy",{w:1360,shade:TEAL,bold:true,color:"FFFFFF",size:18,align:AlignmentType.CENTER}),
    cell("Safe?",{w:1200,shade:TEAL,bold:true,color:"FFFFFF",size:18,align:AlignmentType.CENTER}),
  ]})
];
rays.forEach((ray,i)=>{
  const shade=i%2===0?"FFFFFF":GRAY;
  const opt=optimal[ray];
  const st=opt.stats;
  const safe=opt.safety_ok;
  summaryRows.push(new TableRow({children:[
    cell(ray,{w:1500,shade}),
    cell(String(opt.optimal_count),{w:1500,shade,align:AlignmentType.CENTER,bold:true}),
    cell(st.core_MeV.mean.toLocaleString('en-US',{maximumFractionDigits:2}),{w:2000,shade,align:AlignmentType.CENTER}),
    cell(st.efficiency.toFixed(5),{w:1800,shade,align:AlignmentType.CENTER}),
    cell(opt.dose_gy.toExponential(3),{w:1360,shade,align:AlignmentType.CENTER}),
    cell(safe?"SAFE":"UNSAFE",{w:1200,shade,align:AlignmentType.CENTER,
      bold:true,color:safe?"1A5C2A":"880000"}),
  ]}));
});

const children=[
  // Title
  new Paragraph({spacing:{before:0,after:80},
    children:[new TextRun({text:"STEP 2 — INTERPOLATION SEARCH RESULTS",
      font:"Arial",size:48,bold:true,color:TEAL})]}),
  new Paragraph({spacing:{before:0,after:60},
    children:[new TextRun({text:"Multi-Modal Radiation Therapy for Prion Aggregate Disruption",
      font:"Arial",size:24,bold:true})]}),
  new Paragraph({border:{bottom:{style:BorderStyle.SINGLE,size:8,color:TEAL,space:2}},
    spacing:{before:0,after:240},
    children:[new TextRun({text:`Pure interpolation search — no preset counts — algorithm picks every test point | Run: 2026-03-08`,
      font:"Arial",size:18,italic:true,color:"555555"})]}),

  h1("How the Search Worked"),
  p("Step 2 used a pure iterative interpolation algorithm — not a preset list of example counts. Here is exactly what it did:",{after:80}),
  p("1.  Three anchor points were placed logarithmically across the full search range (50 – 15,000 particles). These established the initial curve shape.",{after:40}),
  p("2.  A quadratic (y = ax² + bx + c) was fitted to all measured points using least-squares Gaussian elimination.",{after:40}),
  p("3.  The predicted peak was computed: x_peak = -b / (2a). The algorithm then fired gears at exactly that count.",{after:40}),
  p("4.  The new data point was added and the quadratic was refit. Step 3 repeated.",{after:40}),
  p("5.  The algorithm stopped when the predicted peak moved less than 30 particles for 2 consecutive iterations — meaning the curve had converged and additional measurements would not change the answer.",{after:160}),
  p("Every test point after the 3 anchors was chosen entirely by the algorithm. The optimal counts below could be any number — they are determined by the physics of each particle type in brain tissue.",{italic:true,color:"555555",after:200}),

  h1("Summary of Results"),
  new Table({width:{size:9360,type:WidthType.DXA},
    columnWidths:[1500,1500,2000,1800,1360,1200],rows:summaryRows}),
  new Paragraph({spacing:{before:240,after:0},children:[]}),

  h1("Overview: All Four Rays"),
  p("The chart below shows the efficiency curves for all four particle types on a single figure. Stars mark the optimal count for each ray.",{after:120}),
  imgPara("plots/step2_all_rays_overview.png", 620*9144, 387*9144),
  new Paragraph({spacing:{before:240,after:0},children:[]}),
];

// Per-ray detail sections
rays.forEach(ray=>{
  const imgName=`step2_${ray.replace(/ /g,'_').toLowerCase()}_curve.png`;
  const opt=optimal[ray];
  const st=opt.stats;
  const rdata=search[ray];
  const hist=rdata.search_history||[];
  const converged=hist.some(h=>h.converged);
  const nPts=Object.keys(rdata.tested).length;

  children.push(
    h1(`${ray} — Detailed Results`),
    new Table({width:{size:9360,type:WidthType.DXA},columnWidths:[2400,2400,2400,2160],rows:[
      new TableRow({children:[
        cell("Optimal Count",{w:2400,shade:TEAL,bold:true,color:"FFFFFF",size:18}),
        cell("Core MeV (mean)",{w:2400,shade:TEAL,bold:true,color:"FFFFFF",size:18}),
        cell("Efficiency",{w:2400,shade:TEAL,bold:true,color:"FFFFFF",size:18}),
        cell("Converged?",{w:2160,shade:TEAL,bold:true,color:"FFFFFF",size:18}),
      ]}),
      new TableRow({children:[
        cell(String(opt.optimal_count),{w:2400,shade:"E8F5E9",bold:true,size:24,align:AlignmentType.CENTER}),
        cell(st.core_MeV.mean.toLocaleString('en-US',{maximumFractionDigits:2})+" MeV",{w:2400,shade:"E8F5E9",size:20,align:AlignmentType.CENTER}),
        cell(st.efficiency.toFixed(5)+" MeV/1kP",{w:2400,shade:"E8F5E9",size:20,align:AlignmentType.CENTER}),
        cell(converged?"YES — algorithm converged":"MAX ITERS reached",
          {w:2160,shade:"E8F5E9",size:18,align:AlignmentType.CENTER,
           color:converged?"1A5C2A":"B8860B",bold:true}),
      ]})
    ]}),
    new Paragraph({spacing:{before:80,after:80},children:[]}),
    p(`Points tested: ${nPts}  |  `+
      `Approx dose: ${opt.dose_gy.toExponential(3)} Gy  |  `+
      `Brain limit: ${({Gamma:24,Neutron:14,"Carbon Ion":20,Alpha:20})[ray]} Gy  |  `+
      `Safety: ${opt.safety_ok?"SAFE":"UNSAFE"}`,
      {italic:true,color:"555555",after:120}),
    p("Search path (predicted peaks per iteration):",{bold:true,after:40}),
    p(hist.length>0
      ? hist.map(h=>`Iter ${h.iter}: predicted ${h.predicted.toFixed(0)} → snapped ${h.snapped}${h.converged?" ✓ CONVERGED":""}`)
           .join("\n")
      : "No search history recorded.",
      {size:18,color:"444444",after:160}),
    imgPara(`plots/${imgName}`, 620*9144, 390*9144),
    new Paragraph({spacing:{before:280,after:0},children:[]}),
  );
});

// What this means section
children.push(
  h1("What These Results Mean"),
  h2("Interpreting Efficiency"),
  p("Efficiency = core MeV deposited per 1000 particles fired. It measures how much therapeutic energy reaches the prion protein core per unit dose delivered. A higher efficiency means you can achieve the same core disruption with fewer particles — important for staying within brain dose limits.",{after:120}),
  h2("Why Different Rays Have Different Optimal Counts"),
  p("Each particle type has a different physical interaction profile in tissue. Carbon Ion has a Bragg peak — its energy deposition is concentrated at a specific depth. This means efficiency rises then falls sharply with particle count as the Bragg peak shifts. Alpha has extremely short range, so increasing count rapidly saturates the core zone. Gamma and Neutron have broader, more stochastic deposition — their efficiency curves are flatter and the interpolation takes more iterations to converge.",{after:120}),
  h2("Dose Safety Notes"),
  p("Dose estimates are approximations: Dose(Gy) = core MeV × 1.602×10⁻¹³ J/MeV ÷ 1.35×10⁻³ kg core mass. Real clinical dosimetry requires fluence, mass-attenuation coefficients and RBE weighting. These values establish order-of-magnitude — not definitive clinical values. The safety flag is conservative.",{after:200,italic:true,color:"666666"}),
);

const doc=new Document({
  styles:{
    default:{document:{run:{font:"Arial",size:22}}},
    paragraphStyles:[
      {id:"Heading1",name:"Heading 1",basedOn:"Normal",next:"Normal",quickFormat:true,
       run:{size:32,bold:true,font:"Arial",color:TEAL},
       paragraph:{spacing:{before:280,after:140},outlineLevel:0}},
      {id:"Heading2",name:"Heading 2",basedOn:"Normal",next:"Normal",quickFormat:true,
       run:{size:26,bold:true,font:"Arial",color:GOLD},
       paragraph:{spacing:{before:200,after:100},outlineLevel:1}},
    ]
  },
  sections:[{
    properties:{page:{size:{width:12240,height:15840},
      margin:{top:1080,right:1080,bottom:1080,left:1080}}},
    headers:{default:new Header({children:[new Paragraph({
      border:{bottom:{style:BorderStyle.SINGLE,size:6,color:TEAL,space:1}},
      spacing:{before:0,after:100},
      children:[
        new TextRun({text:"STEP 2 RESULTS — Interpolation Search",font:"Arial",size:16,bold:true,color:TEAL}),
        new TextRun({text:"  |  Prion Radiation Research  |  ISEF/SAC  |  AI: Claude (Anthropic)",font:"Arial",size:16,color:"888888"}),
      ]})]}),
    },
    footers:{default:new Footer({children:[new Paragraph({
      border:{top:{style:BorderStyle.SINGLE,size:4,color:"CCCCCC",space:1}},
      spacing:{before:100,after:0},alignment:AlignmentType.CENTER,
      children:[
        new TextRun({text:"Page ",font:"Arial",size:16,color:"888888"}),
        new PageNumberElement(),
      ]})]}),
    },
    children,
  }]
});

Packer.toBuffer(doc).then(buf=>{
  fs.writeFileSync("results/step2_report.docx",buf);
  console.log("DONE: results/step2_report.docx");
});
"""
    os.makedirs("results", exist_ok=True)
    with open("_step2_make_doc.js","w") as f: f.write(js)
    env=os.environ.copy()
    env["NODE_PATH"]=os.path.expanduser("~/.npm-global/lib/node_modules")
    r=subprocess.run(["node","_step2_make_doc.js"],capture_output=True,text=True,env=env)
    os.remove("_step2_make_doc.js")
    if r.returncode==0:
        print("\n  Word doc: results/step2_report.docx")
    else:
        print(f"  Word doc error: {r.stderr[:300]}")
else:
    print("  Node not found — skipping Word doc. Plots saved to plots/")

print("\n=== STEP 2 VISUALIZATION COMPLETE ===")
print(f"  Plots saved: {PLOT_DIR}/")
for ray,path in plot_paths.items():
    print(f"    {path}")
print("\nTo view in Finder:  open plots/")
print("Word doc:           open results/step2_report.docx")
