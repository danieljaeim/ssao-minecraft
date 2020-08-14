#version 120

uniform sampler2D lightmap;
uniform sampler2D texture;
uniform sampler2D gdepth;
uniform sampler2D gdepthtex;
uniform sampler2D depthtex0;

varying vec2 lmcoord;
varying vec2 texcoord;
varying vec4 glcolor;
varying vec3 normal;

void main() {
	vec4 color = texture2D(texture, texcoord) * glcolor;
	color *= texture2D(lightmap, lmcoord);
	// vec3 depth = texture2D(depthtex0, texcoord.st).rgb;

/* DRAWBUFFERS:012 */
	gl_FragData[0] = color; //gcolor
	gl_FragData[1] = vec4(normal, 1.0); // color fragments based normal in screen space
	// gl_FragData[2] = vec4(depth, 1.0);
}