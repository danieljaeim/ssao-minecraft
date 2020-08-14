#version 120

const int RGBA16 		= 1;
const int gcolorFormat 	= RGBA16; // turns image into 16 bit

uniform sampler2D gcolor;
uniform sampler2D gnormal; 
uniform sampler2D gdepth;

uniform sampler2D shadow; //gets depth from sun to closest pixel to sun.
uniform sampler2D gdepthtex; //returns the depth buffer from the player's camera
uniform vec3 cameraPosition; // indicates position in world space of player

uniform mat4 gbufferModelViewInverse; // 
uniform mat4 gbufferProjectionInverse;

uniform mat4 shadowModelView;
uniform mat4 shadowProjection;

// shadows, lighting, finalize normal maps, raytracing here
varying vec4 texcoord;
varying vec3 lightPosition;

// we need to get pixels from screen coordinates to world coordinates
float getDepth(in vec2 coord) {
	return texture2D(gdepthtex, coord).r;
}

vec4 getCameraSpacePosition(in vec2 coord) { // theses are screen space coordinates (textcoord.st)
	float depth = getDepth(coord);
	vec4 positionNdcSpace = vec4(coord.s * 2.0 - 1.0, coord.t * 2.0 - 1.0, 2.0 * depth - 1.0, 1.0);
	vec4 positionCameraSpace = gbufferProjectionInverse * positionNdcSpace; //whenever you multiply by matrix, divide by .w
	return positionCameraSpace / positionCameraSpace.w;
}

vec4 getWorldSpacePosition(in vec2 coord) {
	vec4 positionCameraSpace = getCameraSpacePosition(coord);
	vec4 positionWorldSpace = gbufferModelViewInverse * positionCameraSpace;
	positionWorldSpace.xyz += cameraPosition;
	return positionWorldSpace;
}

vec3 getShadowSpacePosition(in vec2 coord) {
	vec4 positionWorldSpace = getWorldSpacePosition(coord);
	positionWorldSpace.xyz -= cameraPosition;
	vec4 positionShadowSpace = shadowModelView * positionWorldSpace;
	positionShadowSpace *= shadowProjection;
	positionShadowSpace /= positionShadowSpace.w;

	return positionShadowSpace.xyz * 0.5 + 0.5;
}

float getSunVisibility(in vec2 coord) { //returns whether in shadow or not
	vec3 shadowCoord = getShadowSpacePosition(coord);
	float shadowMapSample = texture2D(shadow, shadowCoord.st).r;
	return step(shadowCoord.z - shadowMapSample, 0.005);
}

void main() {
	vec3 color = texture2D(gcolor, texcoord.st).rgb;
	vec3 finalNormal = texture2D(gnormal, texcoord.st).rgb;

	// color *= getSunVisibility(texcoord.st);

/* DRAWBUFFERS:0 */
	gl_FragColor = vec4(color, 1.0);
}