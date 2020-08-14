#version 120

uniform int worldTime;
uniform vec3 moonPosition;
uniform vec3 sunPosition;

uniform sampler2D gdepth;

varying vec4 texcoord;
varying vec3 lightPosition;
// varying vec3 normal;
varying vec3 verPos;

void main() {
	gl_Position = ftransform(); // we can access the position of the vertex, after it has been projected to the screen
	texcoord = gl_MultiTexCoord0;
	// normal = normalize(gl_NormalMatrix * gl_Normal);

	// if (worldTime < 12700 || worldTime > 23250) {
	// 	lightPosition = normalize(sunPosition);
	// } else {
	// 	lightPosition = normalize(moonPosition);
	// }
}