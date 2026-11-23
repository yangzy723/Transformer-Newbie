"""按PPT相同的线宽和箭头几何导出矢量PDF。"""
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
from matplotlib import rc_context


def save_vector_pdf(canvases,destination):
    with rc_context({'pdf.fonttype':42,'pdf.compression':6}),PdfPages(destination) as pdf:
        for canvas in canvases:
            fig=Figure(figsize=(13.333333,7.5),dpi=240)
            ax=fig.add_axes([0,0,1,1],xlim=(0,13.333333),ylim=(7.5,0));ax.axis('off')
            for kind,args in canvas.pdf_ops:
                if kind=='rect':
                    x,y,w,h,color,rounded=args
                    patch=(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=0.035',
                        facecolor='#'+color,edgecolor='none') if rounded else
                        Rectangle((x,y),w,h,facecolor='#'+color,edgecolor='none'))
                    ax.add_patch(patch)
                elif kind=='line':
                    x1,y1,x2,y2,color,_,width=args
                    ax.plot([x1,x2],[y1,y2],color='#'+color,lw=width,solid_capstyle='round')
                elif kind=='polygon':
                    points,color=args;ax.add_patch(Polygon(points,facecolor='#'+color,edgecolor='none'))
                elif kind=='text':
                    x,y,line,size,color,path=args
                    ax.text(x,y,line,fontsize=size,fontproperties=FontProperties(fname=path),
                            color='#'+color,va='baseline',parse_math=False)
                elif kind=='image':
                    source,x,y,w,h=args
                    ax.imshow(source,extent=(x,x+w,y+h,y),aspect='auto',interpolation='lanczos')
            for order,artist in enumerate(ax.get_children()):artist.set_zorder(order)
            pdf.savefig(fig);fig.clear()
