#version 120

const int RGBA16 = 1;
const int gcolorformat = RGBA16;

uniform sampler2D gcolor; // given from openGL (2d image of what minecraft samples from)
uniform sampler2D gdepth;
uniform sampler2D gnormal;
uniform sampler2D gdepthtex;
uniform sampler2D noisetex;
uniform sampler2D depthtex0;

uniform sampler2D colortex0; // color
uniform sampler2D colortex1; // normals
uniform sampler2D colortex2; // depth ?
// uniform sampler2D colortex3; // depth ?

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferProjection;

uniform float viewHeight;
uniform float viewWidth;
uniform float near;
uniform vec3 sunPosition;
uniform int worldTime;

varying vec2 texcoord;
varying vec3 normal;

// float GetDistance (in veﬁ››c2 texCoord)
// {
//     const vec4 bitSh = vec4(1.0 / 16777216.0, 1.0 / 65535.0, 1.0 / 256.0, 1.0);
//     return dot(texture2D(gdepthtex, texCoord.st), bitSh) * 1;
// }

// float noisein3(vec3 p){
//     vec3 a = floor(p);
//     vec3 d = p - a;
//     d = d * d * (3.0 - 2.0 * d);

//     vec4 b = a.xxyy + vec4(0.0, 1.0, 0.0, 1.0);
//     vec4 k1 = perm(b.xyxy);
//     vec4 k2 = perm(k1.xyxy + b.zzww);

//     vec4 c = k2 + a.zzzz;
//     vec4 k3 = perm(c);
//     vec4 k4 = perm(c + 1.0);

//     vec4 o1 = fract(k3 * (1.0 / 41.0));
//     vec4 o2 = fract(k4 * (1.0 / 41.0));

//     vec4 o3 = o2 * d.z + o1 * (1.0 - d.z);
//     vec2 o4 = o3.yw * d.x + o3.xz * (1.0 - d.x);

//     return o4.y * d.y + o4.x * (1.0 - d.y);
// }

float randfloat(float n){return fract(sin(n) * 43758.5453123);}

float hash(float n) { return fract(sin(n) * 1e4); }
float hash(vec2 p) { return fract(1e4 * sin(17.0 * p.x + p.y * 0.1) * (0.1 + abs(sin(p.y * 13.0 + p.x)))); }

float noiseintwo(vec2 x) {
	vec2 i = floor(x);
	vec2 f = fract(x);

	// Four corners in 2D of a tile
	float a = hash(i);
	float b = hash(i + vec2(1.0, 0.0));
	float c = hash(i + vec2(0.0, 1.0));
	float d = hash(i + vec2(1.0, 1.0));

	// Simple 2D lerp using smoothstep envelope between the values.
	// return vec3(mix(mix(a, b, smoothstep(0.0, 1.0, f.x)),
	//			mix(c, d, smoothstep(0.0, 1.0, f.x)),
	//			smoothstep(0.0, 1.0, f.y)));

	// Same code, with the clamps in smoothstep and common subexpressions
	// optimized away.
	vec2 u = f * f * (3.0 - 2.0 * f);
	return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

float mod289(float x){return x - floor(x * (1.0 / 289.0)) * 289.0;}
vec4 mod289(vec4 x){return x - floor(x * (1.0 / 289.0)) * 289.0;}
vec4 perm(vec4 x){return mod289(((x * 34.0) + 1.0) * x);}

float noisetothree(vec3 p){
    vec3 a = floor(p);
    vec3 d = p - a;
    d = d * d * (3.0 - 2.0 * d);

    vec4 b = a.xxyy + vec4(0.0, 1.0, 0.0, 1.0);
    vec4 k1 = perm(b.xyxy);
    vec4 k2 = perm(k1.xyxy + b.zzww);

    vec4 c = k2 + a.zzzz;
    vec4 k3 = perm(c);
    vec4 k4 = perm(c + 1.0);

    vec4 o1 = fract(k3 * (1.0 / 41.0));
    vec4 o2 = fract(k4 * (1.0 / 41.0));

    vec4 o3 = o2 * d.z + o1 * (1.0 - d.z);
    vec2 o4 = o3.yw * d.x + o3.xz * (1.0 - d.x);

    return o4.y * d.y + o4.x * (1.0 - d.y);
}

vec3 getNormal(vec2 coord) {
	return normalize(texture2D(colortex1, coord.st).xyz * 2.0 - 1.0);
}

float getDepth(in vec2 coord) {
	return texture2D(depthtex0, coord).x;
}

vec2 getRandom(in vec2 coord) {
	return normalize(texture2D(colortex1, vec2(viewWidth, viewHeight) * coord.st / 4096.0).xy * 2.0f - 1.0f);
}

vec3 getViewSpacePosition(in vec2 uv) {
    float depth = texture2D(depthtex0, uv).x;
    vec4 position = gbufferProjectionInverse * vec4(uv.s * 2.0 - 1.0, uv.t * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    return position.xyz / position.w;
}

// vec3 getRandomKernel(in float x, in float y) {
// 	vec3 random = vec3(rand(vec2(-1.0f, 1.0f)), rand(vec2(-1.0f, 1.0f)), rand(vec2(0.0f, 1.0f)));
// 	random = normalize(random);
// 	float scale = y / x;
// 	random *= 0.1f * (1 - scale * scale) + 1.0f * scale * scale;
// 	return random;
// }

float ambientOcclusionCalc(in vec2 coord, in vec3 vec, in vec3 o, in vec3 norm) {
	float g_bias = 5.5;
	float g_intensity = .0255;	
	vec3 diff = getViewSpacePosition(coord.st + vec.xy) - o;
	vec3 v = normalize(diff);
	float d = length(diff) * 1.0f;
	return max(0.0, dot(norm, v) - g_bias) * (1.0 / (1.0 + d)) * g_intensity;
}

vec2 noiseScale = vec2(viewHeight / 4.0, viewWidth / 4.0);
float kernelSize = 64;

void main() {
	float bias = 0.0025f;
	vec3 color = texture2D(colortex0, texcoord).rgb;
	vec3 position = getViewSpacePosition(texcoord); //fragment pixel position
	vec3 texnormal = getNormal(texcoord); // fragment normal 
	float radius = 10;

	float theta_n = 2 * 3.14 * noisetothree(vec3(texcoord.x, texcoord.y, hash(texcoord.x) * randfloat(texcoord.y)));
	float phi_n = acos(1 - 2 * noisetothree(vec3(texcoord.y, texcoord.x, randfloat(texcoord.x) * hash(texcoord.y))));
	float dir = sin(phi_n) * cos(theta_n);
	float zdir = cos(phi_n);

	// vec3 rvec = normalize(vec3(dir, dir, 0.0) * 2.0 - 1.0); // noise vector attached to tbn
	// vec3 tangent = normalize(rvec - texnormal * dot(rvec, texnormal));
	// vec3 bitangent = cross(texnormal, tangent);
	// mat3 tbn = mat3(tangent, bitangent, texnormal);

	float ao = 0.0f;

	for (float i = 0; i < kernelSize; i++) {
		float theta = 2 * 3.14 * noisetothree(vec3(texcoord.x, texcoord.y, i));
		float phi = acos(1 - 2 * noisetothree(vec3(texcoord.y, texcoord.x, i)));
		float x = sin(phi) * cos(theta);
		float y = sin(phi) * sin(theta);
		float z = cos(phi);

		vec3 sample = normalize(vec3(x, y, z) * 2.0 - 1.0);
		// sample = tbn * sample;
		sample = reflect(sample, texnormal);
		sample = normalize(sample * radius + position);

		vec4 offset = vec4(sample, 1.0f);
		offset = gbufferProjection * offset;
		offset.xyz /= offset.w;
		offset.xyz = offset.xyz * 0.5 + 0.5;

		float sampleDepth = getDepth(texcoord);
		float rangeCheck = smoothstep(0.0, 1.0, radius / abs(position.z - sampleDepth));
		ao += (sampleDepth >= sample.z + bias ? 1.0 : 0.0) * rangeCheck;
	}
	ao = 1 - (ao / kernelSize);
	color -= ao;
	vec3 albedo = vec3(1.0f);
	albedo -= ao;

	// gl_FragData[0] = vec4(albedo, 1.0f);//gcolor
	gl_FragData[0] = vec4(albedo, 1.0f);
	// gl_FragData[0] = vec4(color, 1.0f);

	// // occlusion = 1.0 - (occlusion / kernelSize);
	// // gl_FragData[0] = vec4(texture2D(colortex0, texcoord).rgb, 1.0f);
	// gl_FragData[0] = vec4(color - occlusion, 1.0);

	// vec3 ambient = vec3(0.3 * color * occlusion);
	// vec3 lighting = ambient;
	// vec3 viewDir = normalize(-position);
	// vec3 lightDir = normalize(sunPosition - position); // this is position in view space, sunPosition in worldSpace
	// vec3 diffuse = max(dot(texnormal, lightDir), 0.0) * color * 1.0f;
	// vec3 halfwayDir = normalize(lightDir + viewDir);
	// float spec = pow(max(dot(texnormal, halfwayDir), 0.0), 8.0);
	// vec3 specular = vec3(0.5f) * spec;
	// float dist = length(sunPosition - position);
	// float attenuation = 1.0 / (1.0 + lightLinear * dist + lightQuadratic * dist * dist);
	// diffuse *= attenuation;
	// specular *= attenuation;
	// lighting += diffuse + specular;

	// float focus = 1.5f;
	// float power = 20.0f;
	// float loops = 16.0f;

	// vec3 origin = getViewSpacePosition(texcoord.st); // fragment's view space position
	// vec3 viewnorm = getNormal(texcoord.st);
	// vec2 noise = texture2D(noisetex, texcoord / 32.0).xy + (origin.xy + origin.z);
	// float dist, visibility = 0.0;

	// vec4 random, screenPos, viewPos = vec4(1.0f);

	// for (int i = 0; i < loops; i++) {
	// 	random = texture2D(noisetex, noise);
	// 	random.xyz = random.xyz * 2.0f - 1.0f;
	// 	noise += random.xy;

	// 	if (dot(random.xyz, normal) < 0.0) {
	// 		random.xyz *= -1.0f;
	// 	}

	// 	viewPos.xyz = random.xyz * (focus * random.w) + origin;
	// 	screenPos = gbufferProjection * viewPos;
	// 	dist = GetDistance(screenPos.xy / screenPos.w * 0.5 + 0.5);

	// 	visibility += min(abs((viewPos.z + dist) / focus + 1.0), 1.0);
	// }
/* DRAWBUFFERS:012 */
	// vec4 occlusion = vec4(pow(visibility / loops, 8.0f));
	
	// vec3 albedo = vec3(1.0f);
	// for (int i = 0; i < 4; i++) {
	// 	kernel[i] = vec3(random(-1.0f, 1.0f), random(-1.0f, 1.0f), random(0.0f, 1.0f));
	// 	kernel[i].normalize;
	// 	kernel[i] *= random(0.0f, 1.0f);
	// 	kernel[i] *= lerp(0.1f, 1.0f, 0.125);
	// }

	// float ao = 0.0f;
	// // for (int i = 0; i < 4; i++) {
	// // 	vec3 sample = tbn * kernel[i];
	// // 	sample = sample * 1.0f + origin; 
	// // }

	// float radius = 1.0f / position.z;

	// vec2 example = vec2(1.0f, 0.0f);
	// vec2 example1 = vec2(-1.0f, 0.0f);
	// vec2 example2 = vec2(0.0f, 1.0f);
	// vec2 example3 = vec2(0.0f, -1.0f);

	// // vec2 randomVec = getRandom(texcoord);

	// vec2 coord1 = reflect(example, rand.st) * radius;
	// vec2 coord2 = vec2(coord1.x * 0.707 - coord1.y * 0.707, coord1.x + 0.707 + coord1.y * 0.707);

	// ao += ambientOcclusionCalc(texcoord.st, coord1 * 0.25, position, texnormal); 
	// ao += ambientOcclusionCalc(texcoord.st, coord2 * 0.5, position, texnormal); 
	// ao += ambientOcclusionCalc(texcoord.st, coord1 * 0.75, position, texnormal); 
	// ao += ambientOcclusionCalc(texcoord.st, coord2, position, texnormal); 

	// ao /= 4.0;
	// color -= ao;
// 
	// gl_FragData[0] = vec4(texture2D(depthtex0, texcoord).rgb, 1.0f);//gcolor
}