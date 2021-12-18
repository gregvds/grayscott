# -----------------------------------------------------------------------------
# Copyright (c) 2021 GREGVDS. All rights reserved.
#
# -----------------------------------------------------------------------------

vertex_shader = """
attribute vec2 position;
attribute vec2 texcoord;
varying vec2 v_texcoord;
void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
    v_texcoord = texcoord;
}
"""

render_fragment = """
uniform int pingpong;
uniform int reagent;
uniform sampler2D texture; // u:= r or b following pinpong
uniform sampler2D uscales; // icl, im, icu for U := r,g,b,a
uniform sampler2D vscales; // icl, im, icu for V := r,g,b,a
uniform sampler1D cmap;
varying vec2 v_texcoord;
void main()
{
    float u;
    vec4 uscales2;

    if(pingpong == 0) {
        if(reagent == 1){
            u = texture2D(texture, v_texcoord).r;
            uscales2 = texture2D(uscales, v_texcoord).rgba;
        } else {
            u = texture2D(texture, v_texcoord).g;
            uscales2 = texture2D(vscales, v_texcoord).rgba;
        }
    } else {
        if(reagent == 1){
            u = texture2D(texture, v_texcoord).b;
            uscales2 = texture2D(uscales, v_texcoord).rgba;
        } else {
            u = texture2D(texture, v_texcoord).a;
            uscales2 = texture2D(vscales, v_texcoord).rgba;
        }
    }


    float uicl = uscales2.r;                     // input lower limit for u
    float uim  = uscales2.g;                     // input median for u
    float uicu = uscales2.b;                     // input upper limit for u
    float uibr = uim - uicl;                    // input bottom
    float uiur = uicu - uim;                    // input upper
    float clampu = clamp(u, uicl, uicu);        // clipping
    float uml = 0.5 / uibr;                     // lower-half
    float ubl = 0.0 - uml * uicl;
    float umh = 0.5 / uiur;                     // upper-half
    float ubh = 0.5 - umh * uim;

    float clampedu;
    if(clampu <= uim) {
        clampedu = uml*clampu + ubl;              // remapping lower-half
    } else {
        clampedu = umh*clampu + ubh;              // remapping upper-half
    }

    vec4 color;
    // original no remapping/clamping values
    // color = texture1D(cmap, u);
    // new clamped/remapped values
    color = texture1D(cmap, clampedu);
    gl_FragColor = color;
}
"""

compute_fragment = """
uniform int pingpong;
uniform sampler2D texture; // U,V:= r,g or b,a following pingpong
uniform sampler2D params;  // rU,rV,f,k := r,g,b,a
uniform float dx;          // horizontal distance between texels
uniform float dy;          // vertical distance between texels
uniform float dd;          // unit of distance
uniform float dt;          // unit of time
uniform vec2 brush;        // coordinates of mouse down
varying vec2 v_texcoord;
void main(void)
{
    // Original way of computing the diffusion Laplacian
    // float center = -(4.0+4.0/sqrt(2.0));  // -1 * other weights
    // float diag   = 1.0/sqrt(2.0);         // weight for diagonals
    // float neibor = 1.0;                    // weight for neighbours
    // traditional way of computing diffusion Laplacian
    float center = -1.0;
    float neibor = 0.2;
    float diag = 0.05;
    vec2 p = v_texcoord;                  // center coordinates

    vec2 c,l;
    if( pingpong == 0 ) {
        c = texture2D(texture, p).rg;    // central value
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

    float u = c.r;           // compute some temporary
    float v = c.g;           // values which might save
    float lu = l.r;          // a few GPU cycles
    float lv = l.g;
    float uvv = u * v * v;

    vec4 q = texture2D(params, p).rgba;
    float ru = q.r;          // rate of diffusion of U
    float rv = q.g;          // rate of diffusion of V
    float f  = q.b;          // feed of U
    float k  = q.a;          // kill of V

    float du = ru * lu / dd - uvv + f * (1.0 - u); // Gray-Scott equation
    float dv = rv * lv / dd + uvv - (f + k) * v;   // diffusion+-reaction

    u += du * dt;
    v += dv * dt;

    vec2 diff;
    float dist;
    // Original way of clamping values
    if( pingpong == 1 ) {
        if (brush.x > 0.0) {
            diff = (p - brush)/dx;
            dist = dot(diff, diff);
            if(dist < 3.0) {
                v = 0.5;
            }
        }
        // gl_FragColor = vec4(clamp(u, 0.0, 1.0), clamp(v, 0.0, 1.0), c);
        gl_FragColor = vec4(u, v, c.r, c.g);
    } else {
        if (brush.x > 0.0) {
            diff = (p - brush)/dx;
            dist = dot(diff, diff);
            if(dist < 3.0) {
                v = 0.5;
            }
        }
        // gl_FragColor = vec4(c, clamp(u, 0.0, 1.0), clamp(v, 0.0, 1.0));
        gl_FragColor = vec4(c.r, c.g, u, v);
    }
    // No clamping?
    // if( pingpong == 1 ) {
    //     gl_FragColor = vec4(u, v, c.r, c.g);
    // } else {
    //     gl_FragColor = vec4(c.r, c.g, u, v);
    // }
}
"""
