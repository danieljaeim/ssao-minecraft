#version 120

varying vec2 texcoord;
varying vec4 glcolor;
varying vec2 lmcoord;
varying vec3 normal;

void main() {
	gl_Position = ftransform();
	texcoord = gl_MultiTexCoord0.st;
	glcolor = gl_Color;
	lmcoord  = gl_MultiTexCoord1.st;
	normal = normalize(gl_NormalMatrix * gl_Normal); //assuming this is world Normal
}