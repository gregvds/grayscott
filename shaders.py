# -----------------------------------------------------------------------------
# 2021 GREGVDS
#
# -----------------------------------------------------------------------------

vertex_shader = """
attribute vec2 position;
attribute vec2 texcoord;
varying vec2 highp v_texcoord;
void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
    v_texcoord = texcoord;
}
"""

render_hs_fragment = """
uniform int pingpong;
uniform int reagent;
uniform float hsdir;
uniform float hsalt;
uniform float hsz;
uniform float dx;                // horizontal distance between texels
uniform float dy;                // vertical distance between texels
uniform float dd;                // unit of distance
uniform highp sampler2D texture; // u:= r or b following pinpong
uniform highp sampler2D uscales; // icl, im, icu for U := r,g,b,a
uniform highp sampler2D vscales; // icl, im, icu for V := r,g,b,a
uniform sampler1D cmap;
varying highp vec2 v_texcoord;
void main()
{
    float u;
    vec4 highp scales;
    // pingpong between layers and choice or reagent
    if(pingpong == 0) {
        if(reagent == 1){
            u = texture2D(texture, v_texcoord).r;
            scales = texture2D(uscales, v_texcoord).rgba;
        } else {
            u = texture2D(texture, v_texcoord).g;
            scales = texture2D(vscales, v_texcoord).rgba;
        }
    } else {
        if(reagent == 1){
            u = texture2D(texture, v_texcoord).b;
            scales = texture2D(uscales, v_texcoord).rgba;
        } else {
            u = texture2D(texture, v_texcoord).a;
            scales = texture2D(vscales, v_texcoord).rgba;
        }
    }

    // clamping and rescaling
    float icl = scales.r;                      // input lower limit
    float im  = scales.g;                      // input median
    float icu = scales.b;                      // input upper limit
    float ibr = im - icl;                      // input bottom
    float iur = icu - im;                      // input upper
    float clampu = clamp(u, icl, icu);         // clipping
    float uml = 0.5 / ibr;                     // lower-half
    float ubl = 0.0 - uml * icl;
    float umh = 0.5 / iur;                     // upper-half
    float ubh = 0.5 - umh * im;
    float clampedu;
    if(clampu <= im) {
        clampedu = uml*clampu + ubl;              // remapping lower-half
    } else {
        clampedu = umh*clampu + ubh;              // remapping upper-half
    }

    vec4 color;
    color = texture1D(cmap, u);
    // color = texture1D(cmap, clampedu);

    // Hillshading
    if (hsz > 0.0){
        vec2 highp p = v_texcoord;                  // center coordinates
        vec2 highp dzdx;
        vec2 highp dzdy;
        if( pingpong == 0 ) {
            dzdx = (( texture2D(texture, p + vec2( dx,-dy)).rg
                  + 2*texture2D(texture, p + vec2( dx,0.0)).rg
                  +   texture2D(texture, p + vec2( dx, dy)).rg)
                  - ( texture2D(texture, p + vec2(-dx,-dy)).rg
                  + 2*texture2D(texture, p + vec2(-dx,0.0)).rg
                  +   texture2D(texture, p + vec2(-dx, dy)).rg)) / 8;
            dzdy = (( texture2D(texture, p + vec2(-dx, dy)).rg
                  + 2*texture2D(texture, p + vec2(0.0, dy)).rg
                  +   texture2D(texture, p + vec2( dx, dy)).rg)
                  - ( texture2D(texture, p + vec2(-dx,-dy)).rg
                  + 2*texture2D(texture, p + vec2(0.0,-dy)).rg
                  +   texture2D(texture, p + vec2( dx,-dy)).rg)) / 8;  //
        } else {
            dzdx = (( texture2D(texture, p + vec2( dx,-dy)).ba
                  + 2*texture2D(texture, p + vec2( dx,0.0)).ba
                  +   texture2D(texture, p + vec2( dx, dy)).ba)
                  - ( texture2D(texture, p + vec2(-dx,-dy)).ba
                  + 2*texture2D(texture, p + vec2(-dx,0.0)).ba
                  +   texture2D(texture, p + vec2(-dx, dy)).ba)) / 8;
            dzdy = (( texture2D(texture, p + vec2(-dx, dy)).ba
                  + 2*texture2D(texture, p + vec2(0.0, dy)).ba
                  +   texture2D(texture, p + vec2( dx, dy)).ba)
                  - ( texture2D(texture, p + vec2(-dx,-dy)).ba
                  + 2*texture2D(texture, p + vec2(0.0,-dy)).ba
                  +   texture2D(texture, p + vec2( dx,-dy)).ba)) / 8;  //
        }

        float gradzx, gradzy;
        if(reagent==1){
            gradzx = dzdx.x;
            gradzy = dzdy.x;
        } else {
            gradzx = dzdx.y;
            gradzy = dzdy.y;
        }
        float pi = 3.141592653589793;
        float slope = atan(hsz * sqrt(gradzx*gradzx + gradzy*gradzy));
        float aspect = 0.0;
        if(gradzx != 0.0){
            aspect = atan(gradzy, gradzx);
            if(aspect < 0.0){
                aspect += 2.0*pi;
            }
        } else {
            if(gradzy>0.0){
                aspect = pi/2.0;
            } else if(gradzy<0.0){
                aspect = 2.0*pi - pi/2.0;
            }
        }
        float hs = ((cos(hsalt)*cos(slope)) + (sin(hsalt)*sin(slope))*(cos(hsdir - aspect)));
        vec4 colorhsOverlay;
        if (hs < 0.5){
            colorhsOverlay = 2.0 * color * vec4(hs, hs, hs, 1.0);
        } else {
            colorhsOverlay = 1.0 - 2.0*(1.0-color)*(1.0-vec4(hs, hs, hs, 1.0));
        }
        // vec4 colorhspegtop = (1.0 - 2.0*color)*vec4(hs, hs, hs, 1.0)*vec4(hs, hs, hs, 1.0) + 2.0*vec4(hs, hs, hs, 1.0)*color;
        gl_FragColor = colorhsOverlay;
    } else {
        gl_FragColor = color;
    }

}
"""

compute_fragment = """
uniform int pingpong;
uniform highp sampler2D texture;                           // U,V:= r,g or b,a following pingpong
uniform highp sampler2D params;                            // rU,rV,f,k := r,g,b,a
// RWTexture2DArray<unorm float4> colorTexture;
uniform float dx;                                          // horizontal distance between texels
uniform float dy;                                          // vertical distance between texels
uniform float dd;                                          // unit of distance
uniform float dt;                                          // unit of time
uniform highp vec2 brush;                                  // coordinates of mouse down
uniform int brushtype;
varying highp vec2 v_texcoord;

void main(void)
{
    // Original way of computing the diffusion Laplacian
    float center = -1.0 * sqrt(2.0) * (4.0+4.0/sqrt(2.0)); // -1 * other weights
    float diag   = 1.0;                                    // weight for diagonals
    float neibor = 1.0 * sqrt(2.0);                        // weight for neighbours

    vec2 highp p = v_texcoord;                             // center coordinates
    vec2 highp c;
    vec2 highp l;
    if( pingpong == 0 ) {
        c = texture2D(texture, p).rg;                      // central value
        // Compute Laplacian
        l = ( texture2D(texture, p + vec2(-dx,-dy)).rg
            + texture2D(texture, p + vec2( dx,-dy)).rg
            + texture2D(texture, p + vec2(-dx, dy)).rg
            + texture2D(texture, p + vec2( dx, dy)).rg) * diag
            + ( texture2D(texture, p + vec2(-dx, 0.0)).rg
            + texture2D(texture, p + vec2( dx, 0.0)).rg
            + texture2D(texture, p + vec2(0.0,-dy)).rg
            + texture2D(texture, p + vec2(0.0, dy)).rg) * neibor
            + c * center;
    } else {
        c = texture2D(texture, p).ba;    // central value
        // Compute Laplacian
        l = ( texture2D(texture, p + vec2(-dx,-dy)).ba
            + texture2D(texture, p + vec2( dx,-dy)).ba
            + texture2D(texture, p + vec2(-dx, dy)).ba
            + texture2D(texture, p + vec2( dx, dy)).ba) * diag
            + ( texture2D(texture, p + vec2(-dx, 0.0)).ba
            + texture2D(texture, p + vec2( dx, 0.0)).ba
            + texture2D(texture, p + vec2(0.0,-dy)).ba
            + texture2D(texture, p + vec2(0.0, dy)).ba) * neibor
            + c * center;
    }

    float highp u = c.r;           // compute some temporary
    float highp v = c.g;           // values which might save
    float highp lu = l.r;          // a few GPU cycles
    float highp lv = l.g;
    float highp uvv = u * v * v;

    vec4 highp q = texture2D(params, p).rgba;
    float ru = q.r;          // rate of diffusion of U
    float rv = q.g;          // rate of diffusion of V
    float f  = q.b;          // feed of U
    float k  = q.a;          // kill of V

    // WIP attempt at understanding why diffusion freezes over a given distance
    float weight1 = 1.0;     // Reaction part weight
    float weight2 = 1.0;     // Feed Kill part weight
    float weight3 = 1.0 ;    // Diffusion part weight
    float weight4 = (sqrt(2.0) * (4.0+4.0/sqrt(2.0)));    // Ratio of Diffusion U
    float weight5 = (sqrt(2.0) * (4.0+4.0/sqrt(2.0)));    // Ratio of Diffusion V

    // u + 2v -> v + 2v
    // ru = 1.0
    // rv = 0.5
    // dd = 1.0
    // dt = 1.0
    // f = 0.026
    // k = 0.061

    float highp du = weight3 * (ru * lu / weight4 * dd) - (weight1 * uvv) + weight2 * (f * (1.0 - u));   // Gray-Scott equation
    float highp dv = weight3 * (rv * lv / weight5 * dd) + (weight1 * uvv) - weight2 * ((f + k) * v);   // diffusion+-reaction

    u += du * dt;
    v += dv * dt;

    // Manual mouse feed or kill
    vec2 highp diff;
    float dist;
    if (brush.x > 0.0) {
        diff = (p - brush)/dx;
        dist = dot(diff, diff);
        if((brushtype == 1) && (dist < 3.0))
            v = 0.5;
        if((brushtype == 2) && (dist < 9.0))
            v = 0.0;
    }
    vec4 highp color;
    if( pingpong == 1 ) {
        color = vec4(clamp(u, 0.0, 1.0), clamp(v, 0.0, 1.0), c);
    } else {
        color = vec4(c, clamp(u, 0.0, 1.0), clamp(v, 0.0, 1.0));
    }
    gl_FragColor = color;
}
"""
