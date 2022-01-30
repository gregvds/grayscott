# -----------------------------------------------------------------------------
# 2021 GREGVDS
#
# -----------------------------------------------------------------------------

################################################################################
# 2D shaders (used in gs.py)


vertex_shader = """
// model data
attribute vec2 position;
attribute vec2 texcoord;
// Data (to be interpolated) that is passed on to the fragment shader
varying vec2 highp v_texcoord;

void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
    v_texcoord = texcoord;
}
"""

compute_fragment = """
uniform int pingpong;
uniform highp sampler2D texture;                           // U,V:= r,g or b,a following pingpong
uniform highp sampler2D params;                            // rU,rV,f,k := r,g,b,a
uniform highp sampler2D params2;                           // dX,dY,dD,dT = r,g,b,a
uniform float ddmin;                                       // scaling of dd
uniform float ddmax;                                       // scaling of dd
uniform highp vec2 brush;                                  // coordinates of mouse down
uniform int brushtype;
varying highp vec2 v_texcoord;

void main(void)
{
    // Original way of computing the diffusion Laplacian
    float center = -1.0 * (sqrt(2.0) * 4.0 + 4.0);          // -1 * other weights
    float diag   =  1.0;                                    // weight for diagonals
    float neibor =  1.0 * sqrt(2.0);                        // weight for neighbours

    vec2 highp p = v_texcoord;                              // center coordinates
    vec2 highp c;
    vec2 highp l;

    vec4 parameters2 = texture2D(params2, p).rgba;
    float dx = parameters2.r;                               // usually constant
    float dy = parameters2.g;                               // usually constant
    float dd = parameters2.b*(ddmax-ddmin)+ddmin;           // Can be varrying accross the grid
    float dt = parameters2.a;                               // usually constant.
    // Shows some interaction with dD value, as if dD * dT should not go above a limit

    if( pingpong == 0 ) {
        c = texture2D(texture, p).rg;                       // central value
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
        c = texture2D(texture, p).ba;                       // central value
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

    float highp u = c.r;                                    // compute some temporary
    float highp v = c.g;                                    // values which might save
    float highp lu = l.r;                                   // a few GPU cycles
    float highp lv = l.g;
    float highp uvv = u * v * v;

    vec4 highp q = texture2D(params, p).rgba;
    float ru = q.r;                                         // rate of diffusion of U
    float rv = q.g;                                         // rate of diffusion of V
    float f  = q.b;                                         // feed of U
    float k  = q.a;                                         // kill of V

    // float weight1 = 1.0;     // Reaction part weight
    // float weight2 = 1.0;     // Feed Kill part weight
    // float weight3 = 1.0 ;    // Diffusion part weight
    float weight4 = sqrt(2.0) * 4.0 + 4.0;                  // Ratio of Diffusion U
    float weight5 = sqrt(2.0) * 4.0 + 4.0;                  // Ratio of Diffusion V

    // float highp du = weight3 * (ru * lu / weight4 * dd) - (weight1 * uvv) + weight2 * (f * (1.0 - u));   // Gray-Scott equation
    // float highp dv = weight3 * (rv * lv / weight5 * dd) + (weight1 * uvv) - weight2 * ((f + k) * v);   // diffusion+-reaction
    float highp du = ru * lu / weight4 * dd - uvv + f * (1.0 - u);   // Gray-Scott equation
    float highp dv = rv * lv / weight5 * dd + uvv - (f + k) * v;     // diffusion+-reaction

    u += du * dt;
    v += dv * dt;

    // Manual mouse feed or kill
    vec2 highp diff;
    float dist;

    // allow to force V concentrations locally
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

render_hs_fragment = """
uniform int pingpong;
uniform int reagent;             // toggle render between reagent u and v
uniform float hsdir;             // hillshading lighting direction
uniform float hsalt;             // hillshading lighting altitude
uniform float hsz;               // hillshading 'z' multiplier
uniform float dx;                // horizontal distance between texels
uniform float dy;                // vertical distance between texels
uniform highp sampler2D texture; // u:= r or b following pinpong
uniform sampler1D cmap;          // colormap used to render reagent concentration
uniform sampler1D hscmap;        // colormap used for hillshade lighting
varying highp vec2 v_texcoord;
void main()
{
    float u;
    // pingpong between layers and choice of reagent
    if(pingpong == 0) {
        if(reagent == 1){
            u = texture2D(texture, v_texcoord).r;
        } else {
            u = texture2D(texture, v_texcoord).g;
        }
    } else {
        if(reagent == 1){
            u = texture2D(texture, v_texcoord).b;
        } else {
            u = texture2D(texture, v_texcoord).a;
        }
    }

    // color to render
    vec4 color = texture1D(cmap, u);

    // Hillshading
    if (hsz > 0.0){
        vec2 highp p = v_texcoord;                  // center coordinates
        vec2 highp dzdx;
        vec2 highp dzdy;

        // a b c
        // d e f
        // g h i
        // [dz/dx] = ((c + 2f + i) - (a + 2d + g)) / 8
        // [dz/dy] = ((g + 2h + i) - (a + 2b + c)) / 8
        vec2 highp a;
        vec2 highp b;
        vec2 highp c;
        vec2 highp d;
        vec2 highp f;
        vec2 highp g;
        vec2 highp h;
        vec2 highp i;

        if( pingpong == 0 ) {
            a = texture2D(texture, p + vec2(-dx,-dy)).rg;
            b = texture2D(texture, p + vec2(0.0,-dy)).rg;
            c = texture2D(texture, p + vec2( dx,-dy)).rg;
            d = texture2D(texture, p + vec2(-dx,0.0)).rg;
            f = texture2D(texture, p + vec2( dx,0.0)).rg;
            g = texture2D(texture, p + vec2(-dx, dy)).rg;
            h = texture2D(texture, p + vec2(0.0, dy)).rg;
            i = texture2D(texture, p + vec2( dx, dy)).rg;
        } else {
            a = texture2D(texture, p + vec2(-dx,-dy)).ba;
            b = texture2D(texture, p + vec2(0.0,-dy)).ba;
            c = texture2D(texture, p + vec2( dx,-dy)).ba;
            d = texture2D(texture, p + vec2(-dx,0.0)).ba;
            f = texture2D(texture, p + vec2( dx,0.0)).ba;
            g = texture2D(texture, p + vec2(-dx, dy)).ba;
            h = texture2D(texture, p + vec2(0.0, dy)).ba;
            i = texture2D(texture, p + vec2( dx, dy)).ba;
        }
        dzdx = ((c + 2*f + i) - (a + 2*d + g)) / 8;
        dzdy = ((g + 2*h + i) - (a + 2*b + c)) / 8;

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
        // vec4 hsLux = vec4(hs, hs, hs, 1.0);
        // vec4 hsLux = vec4(hs*(255./256), hs*(229./256), hs*(205./256), 1.0); //warm light
        // vec4 hsLux = vec4(hs*(208./256), hs*(234./256), hs*(255./256), 1.0); //cold light
        vec4 hsLux = texture1D(hscmap, hs);
        vec4 colorhsOverlay;
        if (hs < 0.5){
            // colorhsOverlay = (2.0 * color * hsLux);
            colorhsOverlay = (2.0 * color * hsLux)/sqrt(cos(hsalt))*sin(hsalt);
        } else {
            // colorhsOverlay = (1.0 - 2.0*(1.0-color)*(1.0-hsLux));
            colorhsOverlay = (1.0 - 2.0*(1.0-color)*(1.0-hsLux))/sqrt(cos(hsalt))*sin(hsalt);
        }
        // another way of mixing color and hillshading
        // vec4 colorhspegtop = (1.0 - 2.0*color)*hsLux*hsLux + 2.0*hsLux*color;
        gl_FragColor = colorhsOverlay;
    } else {
        gl_FragColor = color;
    }
}
"""


################################################################################
# 3D shaders (used in gs3D.py and gs3DQt.py)

compute_vertex = """
precision highp float;
precision highp vec2;
precision highp vec3;
precision highp vec4;
precision highp mat4;
precision highp sampler2D;


// model data
attribute vec2 position;
attribute vec2 texcoord;
uniform float dx;                // horizontal distance between texels
uniform float dy;                // vertical distance between texels

// Data (to be interpolated) that is passed on to the fragment shader
varying vec2 v_texcoord;
varying vec2 v_texcoord_diag1;
varying vec2 v_texcoord_diag2;
varying vec2 v_texcoord_diag3;
varying vec2 v_texcoord_diag4;
varying vec2 v_texcoord_neibor1;
varying vec2 v_texcoord_neibor2;
varying vec2 v_texcoord_neibor3;
varying vec2 v_texcoord_neibor4;

void main()
{
    v_texcoord = texcoord;
    v_texcoord_diag1 = texcoord + vec2(-dx,-dy);
    v_texcoord_diag2 = texcoord + vec2( dx,-dy);
    v_texcoord_diag3 = texcoord + vec2(-dx, dy);
    v_texcoord_diag4 = texcoord + vec2( dx, dy);
    v_texcoord_neibor1 = texcoord + vec2(-dx, 0.0);
    v_texcoord_neibor2 = texcoord + vec2( dx, 0.0);
    v_texcoord_neibor3 = texcoord + vec2(0.0,-dy);
    v_texcoord_neibor4 = texcoord + vec2(0.0, dy);
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

compute_fragment_2 = """
precision highp float;
precision highp vec2;
precision highp vec3;
precision highp vec4;
precision highp mat4;
precision highp sampler2D;


uniform int pingpong;
uniform float dx;                // horizontal distance between texels
uniform float dy;                // vertical distance between texels
uniform sampler2D texture; // u:= r or b following pinpong
uniform vec4 params;  // rU,rV,f,k := r,g,b,a
uniform vec2 brush;        // coordinates of mouse down
uniform int brushtype;

// Data coming from the vertex shader
varying vec2 v_texcoord;
varying vec2 v_texcoord_diag1;
varying vec2 v_texcoord_diag2;
varying vec2 v_texcoord_diag3;
varying vec2 v_texcoord_diag4;
varying vec2 v_texcoord_neibor1;
varying vec2 v_texcoord_neibor2;
varying vec2 v_texcoord_neibor3;
varying vec2 v_texcoord_neibor4;

//-------------------------------------------------------------------------

void main()
{
    // Wheights for computing the diffusion Laplacian
    float center = -1.0;                                       // -1 * other weights
    float diag   =  1.0 / (sqrt(2.0) * 4.0 + 4.0);             // weight for diagonals
    float neibor =  1.0 * sqrt(2.0) / (sqrt(2.0) * 4.0 + 4.0); // weight for neighbours

    vec2 p = v_texcoord;                              // center coordinates
    vec2 c;
    vec2 l;

    if( pingpong == 0 ) {
        c = texture2D(texture, p).rg;                       // central value
                                                            // Compute Laplacian
        l = ( texture2D(texture, v_texcoord_diag1).rg
            + texture2D(texture, v_texcoord_diag2).rg
            + texture2D(texture, v_texcoord_diag3).rg
            + texture2D(texture, v_texcoord_diag4).rg) * diag
            + (texture2D(texture, v_texcoord_neibor1).rg
            + texture2D(texture, v_texcoord_neibor2).rg
            + texture2D(texture, v_texcoord_neibor3).rg
            + texture2D(texture, v_texcoord_neibor4).rg) * neibor
            + c * center;
    } else {
        c = texture2D(texture, p).ba;                       // central value
                                                            // Compute Laplacian
        l = ( texture2D(texture, v_texcoord_diag1).ba
            + texture2D(texture, v_texcoord_diag2).ba
            + texture2D(texture, v_texcoord_diag3).ba
            + texture2D(texture, v_texcoord_diag4).ba) * diag
            + (texture2D(texture, v_texcoord_neibor1).ba
            + texture2D(texture, v_texcoord_neibor2).ba
            + texture2D(texture, v_texcoord_neibor3).ba
            + texture2D(texture, v_texcoord_neibor4).ba) * neibor
            + c * center;
    }
    float u = c.r;                                    // compute some temporary
    float v = c.g;                                    // values which might save
    float lu = l.r;                                   // a few GPU cycles
    float lv = l.g;
    float uvv = u * v * v;

    float ru = params.r;
    float rv = params.g;
    float f = params.b;
    float k = params.a;
    // Gray-Scott equation diffusion+-reaction
    // U + 2V -> V + 2V
// For the moment let's put off the dd and dt vars as they are fixed = 1.0
//    float dd = 1.0;
//    float dt = 1.0;
//    float du = ru * lu * dd - uvv + f * (1.0 - u);
//    float dv = rv * lv * dd + uvv - (f + k) * v;
//    u += du * dt;
//    v += dv * dt;
    float du = ru * lu - uvv + f * (1.0 - u);
    float dv = rv * lv + uvv - (f + k) * v;
    u += du;
    v += dv;

    // Manual mouse feed or kill
    vec2 diff;
    float dist;

    // allow to force V concentrations locally
    if (brush.x > 0.0) {
        diff = (p - brush)/dx;
        dist = dot(diff, diff);
        if((brushtype == 1) && (dist < 3.0))
            v = 0.5;
        if((brushtype == 2) && (dist < 9.0))
            v = 0.0;
    }

    vec4 color;
    if( pingpong == 1 ) {
        color = vec4(clamp(u, 0.0, 1.0), clamp(v, 0.0, 1.0), c);
    } else {
        color = vec4(c, clamp(u, 0.0, 1.0), clamp(v, 0.0, 1.0));
    }
    gl_FragColor = color;
}
"""

compute_fragment_3 = """
precision highp float;
precision highp vec2;
precision highp vec3;
precision highp vec4;
precision highp mat4;
precision highp sampler2D;


uniform int pingpong;
uniform float dx;                // horizontal distance between texels
uniform float dy;                // vertical distance between texels
uniform sampler2D texture; // u:= r or b following pinpong
uniform sampler2D params;  // rU,rV,f,k := r,g,b,a
uniform vec2 brush;        // coordinates of mouse down
uniform int brushtype;

// Data coming from the vertex shader
varying vec2 v_texcoord;
varying vec2 v_texcoord_diag1;
varying vec2 v_texcoord_diag2;
varying vec2 v_texcoord_diag3;
varying vec2 v_texcoord_diag4;
varying vec2 v_texcoord_neibor1;
varying vec2 v_texcoord_neibor2;
varying vec2 v_texcoord_neibor3;
varying vec2 v_texcoord_neibor4;

//-------------------------------------------------------------------------

void main()
{
    // Wheights for computing the diffusion Laplacian
    float center = -1.0;                                       // -1 * other weights
    float diag   =  1.0 / (sqrt(2.0) * 4.0 + 4.0);             // weight for diagonals
    float neibor =  1.0 * sqrt(2.0) / (sqrt(2.0) * 4.0 + 4.0); // weight for neighbours

    vec2 p = v_texcoord;                              // center coordinates
    vec2 c;
    vec2 l;

    if( pingpong == 0 ) {
        c = texture2D(texture, p).rg;                       // central value
                                                            // Compute Laplacian
        l = ( texture2D(texture, v_texcoord_diag1).rg
            + texture2D(texture, v_texcoord_diag2).rg
            + texture2D(texture, v_texcoord_diag3).rg
            + texture2D(texture, v_texcoord_diag4).rg) * diag
            + (texture2D(texture, v_texcoord_neibor1).rg
            + texture2D(texture, v_texcoord_neibor2).rg
            + texture2D(texture, v_texcoord_neibor3).rg
            + texture2D(texture, v_texcoord_neibor4).rg) * neibor
            + c * center;
    } else {
        c = texture2D(texture, p).ba;                       // central value
                                                            // Compute Laplacian
        l = ( texture2D(texture, v_texcoord_diag1).ba
            + texture2D(texture, v_texcoord_diag2).ba
            + texture2D(texture, v_texcoord_diag3).ba
            + texture2D(texture, v_texcoord_diag4).ba) * diag
            + (texture2D(texture, v_texcoord_neibor1).ba
            + texture2D(texture, v_texcoord_neibor2).ba
            + texture2D(texture, v_texcoord_neibor3).ba
            + texture2D(texture, v_texcoord_neibor4).ba) * neibor
            + c * center;
    }
    float u = c.r;                                    // compute some temporary
    float v = c.g;                                    // values which might save
    float lu = l.r;                                   // a few GPU cycles
    float lv = l.g;
    float uvv = u * v * v;

    vec4 rurvfk = texture2D(params, p);
    float ru = rurvfk.r;
    float rv = rurvfk.g;
    float f = rurvfk.b;
    float k = rurvfk.a;
    // Gray-Scott equation diffusion+-reaction
    // U + 2V -> V + 2V
// For the moment let's put off the dd and dt vars as they are fixed = 1.0
//    float dd = 1.0;
//    float dt = 1.0;
//    float du = ru * lu * dd - uvv + f * (1.0 - u);
//    float dv = rv * lv * dd + uvv - (f + k) * v;
//    u += du * dt;
//    v += dv * dt;
    float du = ru * lu - uvv + f * (1.0 - u);
    float dv = rv * lv + uvv - (f + k) * v;
    u += du;
    v += dv;

    // Manual mouse feed or kill
    vec2 diff;
    float dist;

    // allow to force V concentrations locally
    if (brush.x > 0.0) {
        diff = (p - brush)/dx;
        dist = dot(diff, diff);
        if((brushtype == 1) && (dist < 3.0))
            v = 0.5;
        if((brushtype == 2) && (dist < 9.0))
            v = 0.0;
    }

    vec4 color;
    if( pingpong == 1 ) {
        color = vec4(clamp(u, 0.0, 1.0), clamp(v, 0.0, 1.0), c);
    } else {
        color = vec4(c, clamp(u, 0.0, 1.0), clamp(v, 0.0, 1.0));
    }
    gl_FragColor = color;
}
"""

render_3D_vertex = """
precision highp float;
precision highp vec2;
precision highp vec3;
precision highp vec4;
precision highp mat4;
precision highp sampler2D;


// Scene transformations
uniform mat4 u_vm;
uniform mat4 u_pvm;
uniform mat4 u_shadowmap_pvm;

// Model parameters
uniform sampler2D texture;            // u:= r or b following pinpong
uniform lowp int pingpong;
uniform lowp int reagent;             // toggle render between reagent u and v
uniform lowp float scalingFactor;
uniform lowp float dx;                // horizontal distance between texels
uniform lowp float dy;                // vertical distance between texels

// Light model
uniform vec3 u_light_position;

// Original model data
attribute vec3 position;
attribute vec2 texcoord;
attribute vec3 normal;
attribute vec4 color;

// Data (to be interpolated) that is passed on to the fragment shader
varying vec3 v_position;
varying vec3 v_light_position;
varying vec3 v_normal;
varying vec2 v_texcoord;
varying vec4 v_Vertex_relative_to_light;


void main()
{
    // Here height and adjacent height are extracted from texture
    // read neightbor heights using an arbitrary small offset
    // Offset is the step of the 3D gridplane
    vec3 off = vec3(dx, dy, 0.0);
    float c;
    float hL;
    float hR;
    float hD;
    float hU;
    if( pingpong == 0 ) {
        if(reagent == 1){
            c = texture2DLod(texture, texcoord, 0).r;                       // central value
            hL = texture2DLod(texture, texcoord - off.xz, 0).r;
            hR = texture2DLod(texture, texcoord + off.xz, 0).r;
            hD = texture2DLod(texture, texcoord - off.zy, 0).r;
            hU = texture2DLod(texture, texcoord + off.zy, 0).r;
        } else {
            c = texture2DLod(texture, texcoord, 0).g;                       // central value
            hL = texture2DLod(texture, texcoord - off.xz, 0).g;
            hR = texture2DLod(texture, texcoord + off.xz, 0).g;
            hD = texture2DLod(texture, texcoord - off.zy, 0).g;
            hU = texture2DLod(texture, texcoord + off.zy, 0).g;
        }
    } else {
        if(reagent == 1){
            c = texture2DLod(texture, texcoord, 0).b;                       // central value
            hL = texture2DLod(texture, texcoord - off.xz, 0).b;
            hR = texture2DLod(texture, texcoord + off.xz, 0).b;
            hD = texture2DLod(texture, texcoord - off.zy, 0).b;
            hU = texture2DLod(texture, texcoord + off.zy, 0).b;
        } else {
            c = texture2DLod(texture, texcoord, 0).a;                       // central value
            hL = texture2DLod(texture, texcoord - off.xz, 0).a;
            hR = texture2DLod(texture, texcoord + off.xz, 0).a;
            hD = texture2DLod(texture, texcoord - off.zy, 0).a;
            hU = texture2DLod(texture, texcoord + off.zy, 0).a;
        }
    }
    c = (1.0 - c)/scalingFactor;
    hL = (1.0 - hL)/scalingFactor;
    hR = (1.0 - hR)/scalingFactor;
    hD = (1.0 - hD)/scalingFactor;
    hU = (1.0 - hU)/scalingFactor;

    // A new position vertex is build from the old vertex coordinates and the concentrations
    // rendered by the compute_fragment(2), hence the surface of the gridplane is
    // embossed / displaced
    vec4 position2 = vec4(position.x, c, position.z, 1.0);

    // Perform the model and view transformations on the vertex and pass this
    // location to the fragment shader.
    v_position = vec3(u_vm * position2);

    // Perform the model and view transformations on the light vertex and pass this
    // location to the fragment shader.
    v_light_position = vec3(u_vm * vec4(u_light_position, 1.0));

    // Calculate this vertex's location from the light source. This is
    // used in the fragment shader to determine if fragments receive direct light.
    v_Vertex_relative_to_light = u_shadowmap_pvm * position2;

    // Here since position has been realtime modified, normals have to be computed again
    // Due to the fact the gridplane is facing +y, these are to be put so:
    vec4 normal2;
    normal2.x = (hD - hU)/dy;
    normal2.y = 2.0;
    normal2.z = (hL - hR)/dx;
    normal2.w = 0.0;
    normal2 = normalize(normal2);

    // Perform the model and view transformations on the vertex's normal vector
    // and pass this normal vector to the fragment shader.
    v_normal = vec3(u_vm * normal2);

    // Pass the texcoord to the fragment shader.
    v_texcoord = texcoord;

    // Transform the location of the vertex for the rest of the graphics pipeline
    gl_Position = u_pvm * position2;
}
"""

render_3D_fragment = """
// this to do partial derivative for normal computation
// #extension GL_OES_standard_derivatives : enable
precision highp float;
precision highp vec2;
precision highp vec3;
precision highp vec4;
precision highp mat4;
precision highp sampler2D;

uniform lowp int pingpong;
uniform lowp int reagent;             // toggle render between reagent u and v

// Light model
uniform vec4 u_ambient_color;
uniform vec4 u_diffuse_color;
uniform vec4 u_specular_color;
uniform vec3 u_light_intensity;
uniform float u_ambient_intensity;
uniform float u_diffuse_intensity;
uniform float u_specular_shininess;
uniform float u_attenuation_c1;
uniform float u_attenuation_c2;
uniform float u_attenuation_c3;
uniform float u_lightbox_fresnelexponant;
uniform float u_lightbox_intensity;

uniform sampler2D texture; // u:= r or b following pinpong
uniform sampler1D cmap;          // colormap used to render reagent concentration
uniform samplerCube cubeMap;

uniform sampler2D shadowMap;
uniform lowp float u_shadow_hardtolerance;
uniform lowp float u_shadow_pcftolerance;
uniform lowp float u_shadow_vsfgate;
uniform lowp float u_shadow_pcfspreading;

uniform bool u_ambient_on;
uniform bool u_diffuse_on;
uniform bool u_attenuation_on;
uniform bool u_specular_on;
uniform bool u_shadow_on;
uniform int u_shadow_type;
uniform bool u_lightbox_on;

// Data coming from the vertex shader
varying vec3 v_position;
varying vec3 v_light_position;
varying vec3 v_normal;
varying vec2 v_texcoord;
varying mediump vec4 v_Vertex_relative_to_light;


//-------------------------------------------------------------------------
// attempt at optimizing simple comparisons (ifs) by replacing them with math

float when_eq_float(float x, float y) {
  return 1.0 - abs(sign(x - y));
}

int when_eq_int(int x, int y) {
  return 1 - int(abs(sign(x - y)));
}

float when_neq(float x, float y) {
  return abs(sign(x - y));
}

float when_gt(float x, float y) {
  return max(sign(x - y), 0.0);
}

float when_lt(float x, float y) {
  return max(sign(y - x), 0.0);
}

float when_ge(float x, float y) {
  return 1.0 - when_lt(x, y);
}

float when_le(float x, float y) {
  return 1.0 - when_gt(x, y);
}

//-------------------------------------------------------------------------
// Vector conversions methods

vec3 scale_from_ndc(vec3 vertex) {
    // Convert the the values from Normalized Device Coordinates (range [-1.0,+1.0])
    // to the range [0.0,1.0]. This mapping is done by scaling
    // the values by 0.5, which gives values in the range [-0.5,+0.5] and then
    // shifting the values by +0.5.
    return vertex * 0.5 + 0.5;
}

vec3 persp_divide(vec4 vertex) {
    // The vertex location rendered from the light source is almost in Normalized
    // Device Coordinates (NDC), but the perspective division has not been
    // performed yet. Perform the perspective divide. The (x,y,z) vertex location
    // components are now each in the range [-1.0,+1.0].
    return vertex.xyz / vertex.w;
}

//-------------------------------------------------------------------------
// Determine if this fragment is in a shadow. Returns true or false.

bool in_shadow(void) {
  vec3 vertex_relative_to_light = scale_from_ndc(persp_divide(v_Vertex_relative_to_light));
  float shadowmap_distance = texture2D(shadowMap, vertex_relative_to_light.xy).r;

  // Test the distance between this fragment and the light source as
  // calculated using the shadowmap transformation (vertex_relative_to_light.z) and
  // the smallest distance between the closest fragment to the light source
  // for this location, as stored in the shadowmap. When the closest
  // distance to the light source was saved in the shadowmap, some
  // precision was lost. Therefore we need a small tolerance factor to
  // compensate for the lost precision.
  return bool(when_gt(vertex_relative_to_light.z, shadowmap_distance + u_shadow_hardtolerance));
}

//-------------------------------------------------------------------------

//-------------------------------------------------------------------------
// Determine if this fragment is in a shadow. Returns ratio of visibility.
// Sample the shadowmap 4 times instead of once and modulate the visibility

float pcf(void) {
  vec2 lowp poissonDisk[4] = vec2[](
   vec2( -0.94201624, -0.39906216 ),
   vec2( 0.94558609, -0.76890725 ),
   vec2( -0.094184101, -0.92938870 ),
   vec2( 0.34495938, 0.29387760 )
  );
  float lowp visibility = 1.0;
//  float lowp spreading = 1000;

  vec3 vertex_relative_to_light = scale_from_ndc(persp_divide(v_Vertex_relative_to_light));

  int lowp index;
  for (int i=0; i<4; i++) {
    visibility -= .2 *
        when_lt(texture2D(shadowMap, vertex_relative_to_light.xy + poissonDisk[i]/u_shadow_pcfspreading).r,
                vertex_relative_to_light.z - u_shadow_pcftolerance );
  }
  return visibility;
}
//-------------------------------------------------------------------------

//-------------------------------------------------------------------------
// Variance shadow filter algorithms using moments

float vsf(void)
{
    vec3 vertex_relative_to_light = scale_from_ndc(persp_divide(v_Vertex_relative_to_light));

    // We retrieve the two moments previously stored (depth and depth*depth)
    vec2 moments = texture2D(shadowMap, vertex_relative_to_light.xy).rg;

    // Surface is fully lit. as the current fragment is before the light occluder
    if (vertex_relative_to_light.z <= moments.x)
        return 1.0 ;

    // to avoid out of shadow frustum requests
    if (v_Vertex_relative_to_light.w > 0.0) {
        // The fragment is either in shadow or penumbra. We now use chebyshev's upperBound to check
        // How likely this pixel is to be lit (p_max)
        float variance = moments.y - (moments.x*moments.x);
        variance = max(variance, u_shadow_vsfgate);

        float d = vertex_relative_to_light.z - moments.x;
        float p_max = variance / (variance + d*d);
//        float p_max = 1.0 - ((variance + d*d) / variance);

        return p_max;
    }
    return 1.0;
}

//-------------------------------------------------------------------------


void main()
{
    vec3 to_light;
    vec3 vertex_normal;
    vec3 reflection;
    vec3 to_camera;
    float cos_angle = 0.0;

    vec4 ambient_color = vec4(0, 0, 0, 1);
    vec4 diffuse_color = vec4(0, 0, 0, 1);
    vec4 specular_color = vec4(0, 0, 0, 1);
    vec4 potential_specular_light = vec4(0, 0, 0, 1);
    vec4 reflected_color = vec4(0, 0, 0, 0);

    float light_distance;
    float attenuationFactor = 1.0;
    float visibility = 1.0;

    //--------------------------------------------------------------------------
    // Sampling of the model concentrations that give the surface color

    float u;
    vec4 texValue = texture2D(texture, v_texcoord);
    u = texValue.r * when_eq_int(reagent, 1) * when_eq_int(pingpong, 0) +
        texValue.g * when_eq_int(reagent, 0) * when_eq_int(pingpong, 0) +
        texValue.b * when_eq_int(reagent, 1) * when_eq_int(pingpong, 1) +
        texValue.a * when_eq_int(reagent, 0) * when_eq_int(pingpong, 1);
    vec4 surface_color = texture1D(cmap, u);

    //--------------------------------------------------------------------------
    // Base definition of the colors as modulation of surface color, partial colors and light intensity

    if (u_ambient_on) {
        ambient_color = vec4(u_ambient_intensity * vec3(surface_color), surface_color.a) * u_ambient_color;
    }
    if (u_diffuse_on) {
        diffuse_color = surface_color * u_diffuse_intensity * u_diffuse_color;
    }
    if (u_specular_on) {
        potential_specular_light = vec4(u_light_intensity, 1) * u_specular_color;
    }

    //--------------------------------------------------------------------------
    // shadows computation following several algorithms

    if (u_shadow_on) {
        if (u_shadow_type == 0 && in_shadow()) {
            // simple shadow mapping
            // This fragment only receives ambient light because it is in a shadow.
            gl_FragColor = ambient_color;
            return;
        } else if (u_shadow_type == 1) {
            // Percent shadow mapping through multiple sampling
            // proportion of full light for this fragment
            visibility = pcf();
        } else if (u_shadow_type == 2) {
            // variance shadow mapping with one sampling
            // proportion of full light for this fragment
            visibility = vsf();
        }
    }

    //--------------------------------------------------------------------------
    // Calculate a vector from the fragment location to the light source
    // This will be used for attenuation, diffuse and specular lighting computation
    to_light = v_light_position - v_position;

    // while computing this vector, let's compute its length and the attenuation
    // due to it before normalizing it
    if (u_attenuation_on)
    {
        light_distance = length(to_light);
        attenuationFactor = 1.0/(u_attenuation_c1 + u_attenuation_c2 * light_distance + u_attenuation_c3 * light_distance * light_distance);
    }
    to_light = normalize( to_light );

    // The vertex's normal vector is being interpolated across the primitive
    // which can make it un-normalized. So normalize the vertex's normal vector.
    vertex_normal = normalize( v_normal );

    // Calculate the cosine of the angle between the vertex's normal vector
    // and the vector going to the light.
    // = 1 when vertices parallel,
    // = 0 when vertices perpendicular
    cos_angle = dot(vertex_normal, to_light);

    //--------------------------------------------------------------------------
    if (u_diffuse_on) {
        //cos_angle = clamp(cos_angle, 0.0, 1.0);

        // Scale the color of this fragment based on its angle to the light.
        diffuse_color = vec4(vec3(diffuse_color) * clamp(cos_angle, 0.0, 1.0), diffuse_color.a);
    }

    // Calculate a vector from the fragment location to the camera.
    // The camera is at the origin, so negating the vertex location gives the vector
    to_camera = -1.0 * v_position;
    to_camera = normalize( to_camera );

    //--------------------------------------------------------------------------
    if (u_specular_on) {
        // Calculate the reflection vector
        reflection = 2.0 * cos_angle * vertex_normal - to_light;
        reflection = normalize( reflection );

        // Calculate the cosine of the angle between the reflection vector
        // and the vector going to the camera.
        // = 1 when vertices parallel,
        // = 0 when vertices perpendicular
        cos_angle = dot(reflection, to_camera);
        cos_angle = clamp(cos_angle, 0.0, 1.0);
        cos_angle = pow(cos_angle, u_specular_shininess);

        // The specular color is from the light source, not the object
        if (cos_angle > 0.0) {
            specular_color = potential_specular_light * cos_angle;
            diffuse_color = diffuse_color * (1.0 - cos_angle);
        }
    }

    //--------------------------------------------------------------------------
    float fresnelFactor;
    if (u_lightbox_on) {
        vec3 reflectedDirection = normalize(reflect(to_camera, vertex_normal));
        reflected_color = u_lightbox_intensity * textureCube(cubeMap, -reflectedDirection);

        fresnelFactor = 1.01 - clamp(dot(vertex_normal, to_camera), 0.0, 1.0);
        fresnelFactor = pow(fresnelFactor, u_lightbox_fresnelexponant);
        reflected_color = fresnelFactor * reflected_color;
    }

    //--------------------------------------------------------------------------

    gl_FragColor = reflected_color +
                   ambient_color +
                   visibility * attenuationFactor * (diffuse_color + specular_color);
}
"""

################################################################################
# This vertex shader outputs also the gl_position as w_position to the fragment

shadow_vertex = """
precision highp float;
precision highp vec2;
precision highp vec3;
precision highp vec4;
precision highp mat4;
precision highp sampler2D;


// Scene transformations
uniform mat4 u_vm;
uniform mat4 u_pvm;

// Model parameters
uniform sampler2D texture; // u:= r or b following pinpong
uniform int pingpong;
uniform int reagent;             // toggle render between reagent u and v
uniform float scalingFactor;

// Original model data
attribute vec3 position;
attribute vec2 texcoord;
attribute vec3 normal;
attribute vec4 color;

// Data (to be interpolated) that is passed on to the fragment shader
varying vec4 w_position;

//-------------------------------------------------------------------------
// A way to avoid ifs...

int when_eq_int(int x, int y) {
  return 1 - int(abs(sign(x - y)));
}

//-------------------------------------------------------------------------

void main()
{
    // Here position.y is read from texture
    vec4 textureValues = texture2DLod(texture, texcoord, 0);
    float c = when_eq_int(pingpong, 0) * when_eq_int(reagent, 1) * textureValues.r +
              when_eq_int(pingpong, 0) * when_eq_int(reagent, 0) * textureValues.g +
              when_eq_int(pingpong, 1) * when_eq_int(reagent, 1) * textureValues.b +
              when_eq_int(pingpong, 1) * when_eq_int(reagent, 0) * textureValues.a;

    c = (1.0 - c)/scalingFactor;
    vec3 position2 = vec3(position.x, c, position.z);

    // Export of the gl_position to the fragment to render depth
    w_position = u_pvm * vec4(position2, 1.0);
    // Transform the location of the vertex for the rest of the graphics pipeline
    gl_Position = w_position;
}
"""

shadow_fragment = """
#extension GL_OES_standard_derivatives : enable

precision highp float;
precision highp vec2;
precision highp vec3;
precision highp vec4;
precision highp sampler2D;

// Data coming from the vertex shader
varying vec4 w_position;

void main()
{
    // We render a shadowmap, so we only need to compute depth.
    float depth = w_position.z / w_position.w;
    depth = depth * 0.5 + 0.5;

    float moment1 = depth;
    float moment2 = depth * depth;

    // Adjusting moments (this is sort of bias per pixel) using partial derivative
    float dx = dFdx(depth);
    float dy = dFdy(depth);
    moment2 += 0.25*(dx*dx+dy*dy) ;

    gl_FragColor = vec4(moment1, moment2, 0.0, 0.0);
//    gl_FragColor.r = depth;
}
"""

################################################################################
# 2D line plot shaders (used in gs.py)

lines_vertex = """
attribute vec2 position;
attribute vec4 color;
varying vec4 v_color;
void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
    v_color = color;
}
"""

lines_fragment = """
varying vec4 v_color;
void main()
{
    gl_FragColor = v_color;
}
"""
