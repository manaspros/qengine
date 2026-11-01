import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
BG="#101315"; FG="#e8ebe9"; MUT="#8e9792"; ACC="#3fc6aa"; RIV="#59615d"; ROOF="#e3a54a"; RULE="#2a3033"
plt.rcParams["font.family"]="Noto Sans"
W,H=12,15.6
fig=plt.figure(figsize=(W,H),dpi=100,facecolor=BG)
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis("off")
L=0.9
ax.text(L,14.65,"QWEN3.8-27B  ·  4-BIT  ·  BATCH 1  ·  ONE RTX 5090",color=MUT,fontsize=15,fontweight="bold")
ax.text(L,13.35,"qengine",color=FG,fontsize=74,fontweight="black")
ax.text(L,12.65,"A from-scratch CUDA inference engine. Faster than every engine",color=FG,fontsize=20)
ax.text(L,12.15,"I could run against it on the same GPU, with and without speculation.",color=FG,fontsize=20)

def panel(top,title,unit,rows,vmax,roof=None):
    ax.text(L,top,title,color=FG,fontsize=24,fontweight="bold")
    ax.text(W-L,top,unit,color=MUT,fontsize=14,ha="right")
    x0=3.55; x1=W-L; bh=0.62; y=top-1.05
    if roof:
        rx=x0+(x1-x0)*roof/vmax
        ax.plot([rx,rx],[y-len(rows)*0.92+0.45,top-0.45],ls=(0,(4,3)),color=ROOF,lw=2)
        ax.text(rx-0.08,top-0.38,f"memory-bandwidth limit {roof}",color=ROOF,fontsize=13,ha="right",va="bottom")
    for name,sub,v,us in rows:
        ax.text(L,y+0.06,name,color=ACC if us else FG,fontsize=19,fontweight="bold" if us else "normal",va="center")
        ax.text(L,y-0.3,sub,color=MUT,fontsize=12.5,va="center")
        if v is None:
            ax.text(x0,y,"no MTP mode",color=MUT,fontsize=15,va="center")
        else:
            w=(x1-x0)*v/vmax
            ax.add_patch(patches.FancyBboxPatch((x0,y-bh/2),w,bh,boxstyle="round,pad=0,rounding_size=0.06",fc=ACC if us else RIV,ec="none"))
            ax.text(x0+w-0.15,y,f"{v:.1f}",color=BG if us else FG,fontsize=18,fontweight="bold",ha="right",va="center")
        y-=0.92
    return y

y=panel(11.1,"Plain decode","tokens / second",[
 ("qengine","W4A16, from scratch",111.8,True),
 ("SparkInfer","NVFP4",100.4,False),
 ("llama.cpp","Q4_0",85.8,False),
 ("exllamav3","EXL3 4.0 bpw",80.2,False)],130,roof=124.5)
ax.plot([L,W-L],[y+0.25,y+0.25],color=RULE,lw=1.2)
y=panel(y-0.35,"With MTP speculative decoding","tokens / second",[
 ("qengine","4 draft tokens",244.1,True),
 ("llama.cpp","3 draft tokens",154.8,False),
 ("exllamav3","4 draft tokens",152.1,False),
 ("SparkInfer","",None,False)],260)
ax.plot([L,W-L],[1.25,1.25],color=RULE,lw=1.2)
ax.text(L,0.8,"8 prompts · 1,024 tokens each · median of 5 runs · geometric mean · all engines in one session",color=MUT,fontsize=13.5)
ax.text(L,0.38,"Speculative output is identical to plain greedy output, token for token.",color=MUT,fontsize=13.5)
fig.savefig(__import__("os").path.join(__import__("os").path.dirname(__file__), "qengine-5090.png"),facecolor=BG)
