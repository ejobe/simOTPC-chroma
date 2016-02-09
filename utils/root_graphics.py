import ROOT as root

def graphics_init():
    style = root.TStyle("Modern", "style")
    print "loading root graphics"
    
    style.SetPadTickX(1)
    style.SetPadTickY(1)

    style.SetLabelSize(0.046, "xyz")
    #style.SetOptTitle(0)
    style.SetMarkerSize(0.5)
    style.SetMarkerStyle(8)
    style.SetPalette(1)
  
    #style.SetLabelFont(42,"XYX");
    #style.SetTitleFont(42,"XYZ");
   
    style.SetFrameBorderMode(0)
    style.SetCanvasDefX(200)
    style.SetCanvasBorderMode(0)
    style.SetPadBorderMode(0)
    style.SetPadColor(0)
    #style.SetTitleXSize(0.055);
    #style.SetTitleYSize(0.055);
    style.SetTitleSize(0.05, "xyz");
    style.SetTitleYOffset(1.2)
    style.SetCanvasColor(0)
    style.SetPadLeftMargin(0.12)
    style.SetPadBottomMargin(0.12)

    style.cd()
    return style

if __name__=='__main__':
    graphics_init()


    
