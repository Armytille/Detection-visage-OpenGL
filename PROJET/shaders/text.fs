#version 330
uniform sampler2D myTexture;
uniform int reverse;
in vec2 vsoTexCoord;

out vec4 fragColor;

void main(void) {
  if(reverse == 0)
    fragColor = texture(myTexture, vsoTexCoord);
  else
    fragColor = texture(myTexture, vsoTexCoord * -1);
}
