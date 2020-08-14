#version 120

// varyings are what we use to pass from vertex to fragment shader
varying vec2 texcoord;
varying vec3 normal;
// varying vec3 verPos;

void main() {
	gl_Position = ftransform(); // gives position of current vertice
	texcoord = gl_MultiTexCoord0.st; // texcoord
	normal = normalize(gl_NormalMatrix * gl_Normal);
	// verPos = gl_Position.xyz;
}