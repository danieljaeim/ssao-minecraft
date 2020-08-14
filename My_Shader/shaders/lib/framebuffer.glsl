#define GCOLOR_OUT      gl_FragData[0]
#define GNORMAL_OUT     gl_FragData[2]

// shadows, lighting, finalize normal maps, raytracing here
const int RGBA16 		= 1;
const int gcolorFormat 	= RGBA16; // turns image into 16 bit

uniform sampler2D gcolor;
uniform sampler2D gnormal; 
uniform sampler2D gdepth;

vec3 getAlbedo(in vec2 coord) {
    return texture2D(gcolor, coord.st).rgb;
}

vec3 getNormal(in vec2 coord) {
    return texture2D(gnormal, coord.st).rgb;
}
