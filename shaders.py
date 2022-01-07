# -----------------------------------------------------------------------------
# 2021 GREGVDS
#
# -----------------------------------------------------------------------------

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

# This one is identical to the vertex_shader
compute_vertex = """
// model data
attribute vec2 position;
attribute vec2 texcoord;
// Data (to be interpolated) that is passed on to the fragment shader
varying highp vec2 v_texcoord;

void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
    v_texcoord = texcoord;
}
"""

# This one will/should become identical to the compute_fragment
compute_fragment_2 = """
uniform int pingpong;
uniform float dx;                // horizontal distance between texels
uniform float dy;                // vertical distance between texels
uniform highp sampler2D texture; // u:= r or b following pinpong
uniform highp sampler2D params;  // rU,rV,f,k := r,g,b,a
uniform highp vec2 brush;        // coordinates of mouse down
uniform int brushtype;

// Data coming from the vertex shader
varying highp vec2 v_texcoord;

void main()
{
    // Wheights for computing the diffusion Laplacian
    float center = -1.0 * (sqrt(2.0) * 4.0 + 4.0);          // -1 * other weights
    float diag   =  1.0;                                    // weight for diagonals
    float neibor =  1.0 * sqrt(2.0);                        // weight for neighbours

    vec2 highp p = v_texcoord;                              // center coordinates
    vec2 highp c;
    vec2 highp l;

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
    float dd = 1.0;
    float dt = 1.0;
    float weight4 = sqrt(2.0) * 4.0 + 4.0;                  // Ratio of Diffusion U
    float weight5 = sqrt(2.0) * 4.0 + 4.0;                  // Ratio of Diffusion V
    // Gray-Scott equation diffusion+-reaction
    // U + 2V -> V + 2V
    float highp du = ru * lu / weight4 * dd - uvv + f * (1.0 - u);
    float highp dv = rv * lv / weight5 * dd + uvv - (f + k) * v;
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

render_3D_vertex = """
// Scene transformations
uniform mat4 u_projection;
uniform mat4 u_view;
uniform mat4 u_model;
// uniform mat4 u_Shadowmap_transform; // The pv transform used to render the shadow map
uniform mat4 u_Shadowmap_projection;
uniform mat4 u_Shadowmap_view;

// Model parameters
uniform highp sampler2D texture; // u:= r or b following pinpong
uniform int pingpong;
uniform int reagent;             // toggle render between reagent u and v
uniform highp float scalingFactor;
uniform float dx;                // horizontal distance between texels
uniform float dy;                // vertical distance between texels

// Light model
uniform vec4 u_color;

// Original model data
attribute vec3 position;
attribute vec2 texcoord;
attribute vec3 normal;
attribute vec4 color;

// Data (to be interpolated) that is passed on to the fragment shader
varying vec3 v_position;
varying vec3 v_normal;
varying vec4 v_color;
varying vec2 v_texcoord;
varying vec4 v_Vertex_relative_to_light;

void main()
{
    // Here position.z is received from texture
    // and adjacent position.z are extracted too from texture
    highp vec2 p = texcoord;
    // read neightbor heights using an arbitrary small offset
    // Offset should be 1 / width height
    highp vec3 off = vec3(dx, dy, 0.0);
    highp float c;
    highp float hL;
    highp float hR;
    highp float hD;
    highp float hU;

    if( pingpong == 0 ) {
        if(reagent == 1){
            c = texture2DLod(texture, p, 0).r;                       // central value
            hL = texture2DLod(texture, p - off.xz, 0).r;
            hR = texture2DLod(texture, p + off.xz, 0).r;
            hD = texture2DLod(texture, p - off.zy, 0).r;
            hU = texture2DLod(texture, p + off.zy, 0).r;
        } else {
            c = texture2DLod(texture, p, 0).g;                       // central value
            hL = texture2DLod(texture, p - off.xz, 0).g;
            hR = texture2DLod(texture, p + off.xz, 0).g;
            hD = texture2DLod(texture, p - off.zy, 0).g;
            hU = texture2DLod(texture, p + off.zy, 0).g;
        }
    } else {
        if(reagent == 1){
            c = texture2DLod(texture, p, 0).b;                       // central value
            hL = texture2DLod(texture, p - off.xz, 0).b;
            hR = texture2DLod(texture, p + off.xz, 0).b;
            hD = texture2DLod(texture, p - off.zy, 0).b;
            hU = texture2DLod(texture, p + off.zy, 0).b;
        } else {
            c = texture2DLod(texture, p, 0).a;                       // central value
            hL = texture2DLod(texture, p - off.xz, 0).a;
            hR = texture2DLod(texture, p + off.xz, 0).a;
            hD = texture2DLod(texture, p - off.zy, 0).a;
            hU = texture2DLod(texture, p + off.zy, 0).a;
        }
    }
    c = (1.0 - c)/scalingFactor;
    hL = (1.0 - hL)/scalingFactor;
    hR = (1.0 - hR)/scalingFactor;
    hD = (1.0 - hD)/scalingFactor;
    hU = (1.0 - hU)/scalingFactor;
    highp vec3 position2 = vec3(position.x, position.y, c);

    // Perform the model and view transformations on the vertex and pass this
    // location to the fragment shader.
    v_position = vec3(u_view * u_model * vec4(position2, 1.0));

    // Calculate this vertex's location from the light source. This is
    // used in the fragment shader to determine if fragments receive direct light.
    // v_Vertex_relative_to_light = u_Shadowmap_projection * u_Shadowmap_view * u_model * vec4(position2, 1.0);
    v_Vertex_relative_to_light = u_Shadowmap_projection * u_Shadowmap_view * vec4(position2, 1.0);

    // Here since position has been realtime modified, normals have to be computed again
    highp vec3 N;
    N.x = (hL - hR)/dx;
    N.y = (hD - hU)/dy;
    N.z = 2.0;
    N = normalize(N);

    // Perform the model and view transformations on the vertex's normal vector
    // and pass this normal vector to the fragment shader.
    v_normal = vec3(u_view * u_model * vec4(N, 0.0));

    // Pass the texcoord to the fragment shader.
    v_texcoord = texcoord;

    // Transform the location of the vertex for the rest of the graphics pipeline
    gl_Position = u_projection * u_view * u_model * vec4(position2, 1.0);
}
"""

render_3D_fragment = """
uniform int pingpong;
uniform int reagent;             // toggle render between reagent u and v

// Light model
uniform vec3 u_light_position;
uniform vec3 u_light_intensity;
uniform float u_Shininess;
uniform vec3 u_Ambient_color;
uniform float c1;
uniform float c2;
uniform float c3;

uniform highp sampler2D texture; // u:= r or b following pinpong
uniform highp sampler1D cmap;          // colormap used to render reagent concentration

uniform highp sampler2D shadowMap;
uniform highp float near;
uniform highp float far;
uniform float u_Tolerance_constant;

uniform bool ambient;
uniform bool diffuse;
uniform bool specular;
uniform bool shadow;


// Data coming from the vertex shader
varying vec3 v_position;
varying vec3 v_normal;
varying highp vec2 v_texcoord;
varying vec4 v_Vertex_relative_to_light;

//-------------------------------------------------------------------------
// Determine if this fragment is in a shadow. Returns true or false.
bool in_shadow(void) {

  // The vertex location rendered from the light source is almost in Normalized
  // Device Coordinates (NDC), but the perspective division has not been
  // performed yet. Perform the perspective divide. The (x,y,z) vertex location
  // components are now each in the range [-1.0,+1.0].
  vec3 vertex_relative_to_light = v_Vertex_relative_to_light.xyz / v_Vertex_relative_to_light.w;

  // Convert the the values from Normalized Device Coordinates (range [-1.0,+1.0])
  // to the range [0.0,1.0]. This mapping is done by scaling
  // the values by 0.5, which gives values in the range [-0.5,+0.5] and then
  // shifting the values by +0.5.
  vertex_relative_to_light = vertex_relative_to_light * 0.5 + 0.5;

  // Get the z value of this fragment in relationship to the light source.
  // This value was stored in the shadow map (depth buffer of the frame buffer)
  // which was passed to the shader as a texture map.
  vec4 shadowmap_color = texture2D(shadowMap, vertex_relative_to_light.xy);

  // The texture map contains a single depth value for each pixel. However,
  // the texture2D sampler always returns a color from a texture. For a
  // gl.DEPTH_COMPONENT texture, the color contains the depth value in
  // each of the color components. If the value was d, then the color returned
  // is (d,d,d,1). This is a "color" (depth) value between [0.0,+1.0].
  // float shadowmap_distance = near + (far - near) * shadowmap_color.r;
  float shadowmap_distance = shadowmap_color.r;

  // Test the distance between this fragment and the light source as
  // calculated using the shadowmap transformation (vertex_relative_to_light.z) and
  // the smallest distance between the closest fragment to the light source
  // for this location, as stored in the shadowmap. When the closest
  // distance to the light source was saved in the shadowmap, some
  // precision was lost. Therefore we need a small tolerance factor to
  // compensate for the lost precision.
  if ( vertex_relative_to_light.z <= shadowmap_distance + u_Tolerance_constant ) {
  // if ( vertex_relative_to_light.z <= shadowmap_distance ) {
    // This surface receives full light because it is the closest surface
    // to the light.
    return false;
  } else {
    // This surface is in a shadow because there is a closer surface to
    // the light source.
    return true;
  }
}

//-------------------------------------------------------------------------

void main()
{
    vec3 to_light;
    vec3 vertex_normal;
    vec3 reflection;
    vec3 to_camera;
    float cos_angle;
    vec3 diffuse_color;
    vec3 specular_color;
    vec3 ambient_color;
    vec3 color;
    float light_distance;
    float attenuation;

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
    vec4 v_color = texture1D(cmap, u);

    // Calculate the ambient color as a percentage of the surface color
    if (ambient) {
        ambient_color = u_Ambient_color * vec3(v_color);
    } else {
        ambient_color = vec3(0, 0, 0);
    }

    if (in_shadow() && shadow) {
        // This fragment only receives ambient light because it is in a shadow.
        gl_FragColor = vec4(ambient_color, v_color.a);
        // gl_FragColor = vec4(.1, .1, .1, v_color.a);
        return;
    }

    // Calculate a vector from the fragment location to the light source
    to_light = u_light_position - v_position;

    // while computing this vector, let's compute its length and the attenuation
    // due to it before normalizing it
    light_distance = length(to_light);
    attenuation = 1.0/(c1 + c2 * light_distance + c3 * light_distance * light_distance);
    to_light = normalize( to_light );

    // The vertex's normal vector is being interpolated across the primitive
    // which can make it un-normalized. So normalize the vertex's normal vector.
    vertex_normal = normalize( v_normal );

    // Calculate the cosine of the angle between the vertex's normal vector
    // and the vector going to the light.
    cos_angle = dot(vertex_normal, to_light);
    cos_angle = clamp(cos_angle, 0.0, 1.0);

    // Scale the color of this fragment based on its angle to the light.
    if (diffuse) {
        diffuse_color = vec3(v_color) * cos_angle;
    } else {
        diffuse_color = vec3(0, 0, 0);
    }

    // Calculate the reflection vector
    reflection = 2.0 * dot(vertex_normal,to_light) * vertex_normal - to_light;

    // Calculate a vector from the fragment location to the camera.
    // The camera is at the origin, so negating the vertex location gives the vector
    to_camera = -1.0 * v_position;

    // Calculate the cosine of the angle between the reflection vector
    // and the vector going to the camera.
    reflection = normalize( reflection );
    to_camera = normalize( to_camera );
    cos_angle = dot(reflection, to_camera);
    cos_angle = clamp(cos_angle, 0.0, 1.0);
    cos_angle = pow(cos_angle, u_Shininess);

    // The specular color is from the light source, not the object
    if (cos_angle > 0.0 && specular) {
        specular_color = u_light_intensity * cos_angle;
        diffuse_color = diffuse_color * (1.0 - cos_angle);
    } else {
        specular_color = vec3(0.0, 0.0, 0.0);
    }

    // don't really know on which part of the light sources should the attenuation play
    // Maybe not on the ambient_color?
    color = ambient_color + attenuation * (diffuse_color + specular_color);

    gl_FragColor = vec4(color, v_color.a);
}
"""

# This vertex shader is somehow the same as the render_3D_vertex
shadow_vertex = """
// Scene transformations
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

// Model parameters
uniform highp sampler2D texture; // u:= r or b following pinpong
uniform int pingpong;
uniform int reagent;             // toggle render between reagent u and v
uniform highp float scalingFactor;
uniform float dx;                // horizontal distance between texels
uniform float dy;                // vertical distance between texels

// Light model
uniform vec4 u_color;

// Original model data
attribute vec3 position;
attribute vec2 texcoord;
attribute vec3 normal;
attribute vec4 color;

// Data (to be interpolated) that is passed on to the fragment shader
varying vec3 v_position;
varying vec4 w_position;
varying vec2 v_texcoord;

void main()
{
    // Here position.z is received from texture
    highp vec2 p = texcoord;
    highp float c;

    if( pingpong == 0 ) {
        if(reagent == 1){
            c = texture2DLod(texture, p, 0).r;                       // central value
        } else {
            c = texture2DLod(texture, p, 0).g;                       // central value
        }
    } else {
        if(reagent == 1){
            c = texture2DLod(texture, p, 0).b;                       // central value
        } else {
            c = texture2DLod(texture, p, 0).a;                       // central value
        }
    }
    c = (1.0 - c)/scalingFactor;
    highp vec3 position2 = vec3(position.x, position.y, c);

    // Perform the model and view transformations on the vertex and pass this
    // location to the fragment shader.
    v_position = vec3(u_view * u_model * vec4(position2, 1.0));
    w_position = u_projection * u_view * u_model * vec4(position2, 1.0);
    // w_position = u_view * u_model * vec4(position2, 1.0);

    // Pass the texcoord to the fragment shader.
    v_texcoord = texcoord;

    // Transform the location of the vertex for the rest of the graphics pipeline
    gl_Position = u_projection * u_view * u_model * vec4(position2, 1.0);
}
"""

# this one is currently identical to render_3D_fragment
# but could be simplified as the render color do not interest us, only the
# depth render buffer is important
shadow_fragment = """
uniform int pingpong;
uniform int reagent;             // toggle render between reagent u and v

uniform highp sampler2D texture; // u:= r or b following pinpong
uniform highp float near;
uniform highp float far;

// Data coming from the vertex shader
varying vec3 v_position;
varying vec4 w_position;
varying highp vec2 v_texcoord;

void main()
{
    //float u;
    // pingpong between layers and choice of reagent
    //if(pingpong == 0) {
    //    if(reagent == 1){
    //        u = texture2D(texture, v_texcoord).r;
    //    } else {
    //        u = texture2D(texture, v_texcoord).g;
    //    }
    //} else {
    //    if(reagent == 1){
    //        u = texture2D(texture, v_texcoord).b;
    //    } else {
    //        u = texture2D(texture, v_texcoord).a;
    //    }
    //}

    float depth = w_position.z / w_position.w;
    depth = depth * 0.5 + 0.5;
    gl_FragColor = vec4(depth, depth, depth, 1.0);

    // just output of grayscale to have something to look at from cam
    // gl_FragColor = vec4(u, u, u, 1);
}
"""

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
