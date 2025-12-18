#include "glad.h"
#define SDL_DISABLE_OLD_NAMES
#include <SDL3/SDL.h>
#include <SDL3/SDL_opengl.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>
#include <cmath>

// Embedded Shaders
const char* shadowVertexShader = R"(
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 lightSpaceMatrix;
uniform mat4 model;

void main() {
    gl_Position = lightSpaceMatrix * model * vec4(aPos, 1.0);
}
)";

const char* shadowFragmentShader = R"(
#version 330 core
void main() {
    // gl_FragDepth is automatically written
}
)";

const char* mainVertexShader = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 lightSpaceMatrix;

out vec3 FragPos;
out vec3 Normal;
out vec4 FragPosLightSpace;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    // Transform normal using normal matrix (handles non-uniform scaling)
    mat3 normalMatrix = mat3(transpose(inverse(model)));
    Normal = normalize(normalMatrix * aNormal);
    FragPosLightSpace = lightSpaceMatrix * vec4(FragPos, 1.0);
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
)";

const char* mainFragmentShader = R"(
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec4 FragPosLightSpace;

uniform vec3 lightDir;
uniform vec3 lightColor;
uniform vec3 viewPos;
uniform vec3 objectColor;
uniform sampler2D shadowMap;

float ShadowCalculation(vec4 fragPosLightSpace) {
    // Perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // Transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    
    // Check if coordinates are within shadow map bounds
    // Use >= and <= to include exact boundaries, and check z < 0.0 as well
    if(projCoords.x < 0.0 || projCoords.x > 1.0 || 
       projCoords.y < 0.0 || projCoords.y > 1.0 ||
       projCoords.z < 0.0 || projCoords.z > 1.0)
        return 0.0;
    
    // Get closest depth value from light's perspective
    float closestDepth = texture(shadowMap, projCoords.xy).r;
    // Get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;
    
    // Simple shadow test - if current depth is greater than closest depth, we're in shadow
    // Use a small bias to prevent shadow acne
    float bias = 0.005;
    float shadow = currentDepth > closestDepth + bias ? 1.0 : 0.0;
    
    return shadow;
}

void main() {
    vec3 color = objectColor;
    vec3 normal = normalize(Normal);
    vec3 lightDirection = normalize(-lightDir);
    
    // Ambient - disabled
    vec3 ambient = vec3(0.0);
    
    // Diffuse
    float diff = max(dot(normal, lightDirection), 0.0);
    vec3 diffuse = diff * lightColor;
    
    // Specular
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDirection, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 specular = specularStrength * spec * lightColor;
    
    // Shadow
    float shadow = ShadowCalculation(FragPosLightSpace);
    
    // Calculate final lighting - only diffuse + specular
    vec3 lighting = (1.0 - shadow) * (diffuse + specular) * color;
    
    FragColor = vec4(lighting, 1.0);
}
)";

const char* screenVertexShader = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aTexCoord;
}
)";

const char* screenFragmentShader = R"(
#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D screenTexture;

void main() {
    FragColor = texture(screenTexture, TexCoord);
}
)";

const char* shadowMapVisVertexShader = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aTexCoord;
}
)";

const char* shadowMapVisFragmentShader = R"(
#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D shadowMap;

void main() {
    float depth = texture(shadowMap, TexCoord).r;
    // Visualize depth as grayscale
    FragColor = vec4(vec3(depth), 1.0);
}
)";

// Helper function to compile shader
GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader compilation error: " << infoLog << std::endl;
    }
    return shader;
}

// Helper function to create shader program
GLuint createShaderProgram(const char* vertexSource, const char* fragmentSource) {
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSource);
    
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Shader linking error: " << infoLog << std::endl;
    }
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    return program;
}

// Generate unit cube geometry (1x1x1 centered at origin)
void generateCube(std::vector<float>& vertices, std::vector<unsigned int>& indices) {
    float size = 0.5f;
    vertices = {
        // Positions (x, y, z)          Normals (nx, ny, nz)
        -size, -size, -size,    0.0f, 0.0f, -1.0f,
         size, -size, -size,    0.0f, 0.0f, -1.0f,
         size,  size, -size,    0.0f, 0.0f, -1.0f,
        -size,  size, -size,    0.0f, 0.0f, -1.0f,
        
        -size, -size,  size,    0.0f, 0.0f, 1.0f,
         size, -size,  size,    0.0f, 0.0f, 1.0f,
         size,  size,  size,    0.0f, 0.0f, 1.0f,
        -size,  size,  size,    0.0f, 0.0f, 1.0f,
        
        -size,  size,  size,    -1.0f, 0.0f, 0.0f,
        -size,  size, -size,    -1.0f, 0.0f, 0.0f,
        -size, -size, -size,    -1.0f, 0.0f, 0.0f,
        -size, -size,  size,    -1.0f, 0.0f, 0.0f,
        
         size,  size,  size,     1.0f, 0.0f, 0.0f,
         size,  size, -size,     1.0f, 0.0f, 0.0f,
         size, -size, -size,     1.0f, 0.0f, 0.0f,
         size, -size,  size,     1.0f, 0.0f, 0.0f,
        
        -size, -size, -size,    0.0f, -1.0f, 0.0f,
         size, -size, -size,    0.0f, -1.0f, 0.0f,
         size, -size,  size,    0.0f, -1.0f, 0.0f,
        -size, -size,  size,    0.0f, -1.0f, 0.0f,
        
        -size,  size, -size,    0.0f, 1.0f, 0.0f,
         size,  size, -size,    0.0f, 1.0f, 0.0f,
         size,  size,  size,    0.0f, 1.0f, 0.0f,
        -size,  size,  size,    0.0f, 1.0f, 0.0f,
    };
    
    indices = {
        // Back face (-Z) - CCW when viewed from +Z
        0, 1, 2,  2, 3, 0,
        // Front face (+Z) - CCW when viewed from -Z
        4, 7, 6,  6, 5, 4,
        // Left face (-X) - CCW when viewed from +X
        8, 11, 10,  10, 9, 8,
        // Right face (+X) - CCW when viewed from -X
        12, 13, 14,  14, 15, 12,
        // Bottom face (-Y) - CCW when viewed from +Y
        16, 19, 18,  18, 17, 16,
        // Top face (+Y) - CCW when viewed from -Y
        20, 21, 22,  22, 23, 20
    };
}

// Create VAO from vertices and indices
GLuint createVAO(const std::vector<float>& vertices, const std::vector<unsigned int>& indices) {
    GLuint VAO, VBO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    
    glBindVertexArray(VAO);
    
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);
    
    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glBindVertexArray(0);
    
    return VAO;
}

int main() {
    // Initialize SDL
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::cerr << "SDL initialization failed: " << SDL_GetError() << std::endl;
        return -1;
    }
    
    // Set OpenGL attributes
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    
    // Create window
    SDL_Window* window = SDL_CreateWindow("Shadow Test", 1280, 720, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    if (!window) {
        std::cerr << "Window creation failed: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return -1;
    }
    
    // Create OpenGL context
    SDL_GLContext context = SDL_GL_CreateContext(window);
    if (!context) {
        std::cerr << "OpenGL context creation failed: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }
    
    // Load OpenGL functions
    if (!gladLoadGL()) {
        std::cerr << "Failed to load OpenGL functions" << std::endl;
        SDL_GL_DestroyContext(context);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }
    
    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    
    // Enable depth testing
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CW);
    
    // Create shader programs
    GLuint shadowShader = createShaderProgram(shadowVertexShader, shadowFragmentShader);
    GLuint mainShader = createShaderProgram(mainVertexShader, mainFragmentShader);
    GLuint screenShader = createShaderProgram(screenVertexShader, screenFragmentShader);
    GLuint shadowMapVisShader = createShaderProgram(shadowMapVisVertexShader, shadowMapVisFragmentShader);
    
    // Generate unit cube geometry (used for both plane and cube)
    std::vector<float> cubeVertices;
    std::vector<unsigned int> cubeIndices;
    generateCube(cubeVertices, cubeIndices);
    
    GLuint unitCubeVAO = createVAO(cubeVertices, cubeIndices);
    GLuint cubeIndexCount = cubeIndices.size();
    
    // Create shadow map framebuffer (2048x2048)
    const unsigned int SHADOW_WIDTH = 2048, SHADOW_HEIGHT = 2048;
    GLuint shadowMapFBO, shadowMap;
    glGenFramebuffers(1, &shadowMapFBO);
    glGenTextures(1, &shadowMap);
    glBindTexture(GL_TEXTURE_2D, shadowMap);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float borderColor[] = {1.0f, 1.0f, 1.0f, 1.0f};
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    
    glBindFramebuffer(GL_FRAMEBUFFER, shadowMapFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadowMap, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Shadow framebuffer not complete!" << std::endl;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    // Create offscreen framebuffer (1920x1080)
    const unsigned int OFFSCREEN_WIDTH = 1920, OFFSCREEN_HEIGHT = 1080;
    GLuint offscreenFBO, offscreenColorTexture, offscreenDepthTexture;
    glGenFramebuffers(1, &offscreenFBO);
    glGenTextures(1, &offscreenColorTexture);
    glGenTextures(1, &offscreenDepthTexture);
    
    glBindTexture(GL_TEXTURE_2D, offscreenColorTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, OFFSCREEN_WIDTH, OFFSCREEN_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    glBindTexture(GL_TEXTURE_2D, offscreenDepthTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, OFFSCREEN_WIDTH, OFFSCREEN_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    
    glBindFramebuffer(GL_FRAMEBUFFER, offscreenFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, offscreenColorTexture, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, offscreenDepthTexture, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Offscreen framebuffer not complete!" << std::endl;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    // Create screen quad for rendering final result
    float quadVertices[] = {
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };
    GLuint quadVAO, quadVBO;
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    // Create small quad for shadow map visualization (top right corner)
    // Quad covers from (0.6, 0.6) to (1.0, 1.0) in NDC space
    float shadowVisQuad[] = {
         0.6f,  1.0f,  0.0f, 1.0f,  // Top-left
         0.6f,  0.6f,  0.0f, 0.0f,  // Bottom-left
         1.0f,  0.6f,  1.0f, 0.0f,  // Bottom-right
         0.6f,  1.0f,  0.0f, 1.0f,  // Top-left
         1.0f,  0.6f,  1.0f, 0.0f,  // Bottom-right
         1.0f,  1.0f,  1.0f, 1.0f   // Top-right
    };
    GLuint shadowVisQuadVAO, shadowVisQuadVBO;
    glGenVertexArrays(1, &shadowVisQuadVAO);
    glGenBuffers(1, &shadowVisQuadVBO);
    glBindVertexArray(shadowVisQuadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, shadowVisQuadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(shadowVisQuad), shadowVisQuad, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);
    
    // Light setup - light is directly above origin, pointing down
    glm::vec3 lightDir = glm::vec3(0.0f, -1.0f, 0.0f);
    glm::vec3 lightColor(1.0f, 1.0f, 1.0f);
    
    // Camera setup
    glm::vec3 cameraPos(0.0f, 5.0f, 10.0f);
    glm::vec3 cameraTarget(0.0f, 0.0f, 0.0f);
    glm::vec3 cameraUp(0.0f, 1.0f, 0.0f);
    
    // Main loop
    bool running = true;
    Uint64 startTime = SDL_GetTicks();
    
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) {
                running = false;
            }
        }
        
        float currentTime = (SDL_GetTicks() - startTime) / 1000.0f;
        
        // Calculate model matrices
        glm::mat4 planeModel = glm::scale(
            glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -4.0f, 0.0f)),
            glm::vec3(10.0f, 1.0f, 10.0f)
        );
        glm::mat4 cubeModel = glm::rotate(glm::mat4(1.0f), currentTime, glm::vec3(0.0f, 1.0f, 0.0f));
        
        // Calculate light space matrix for shadow mapping
        float near_plane = 1.0f, far_plane = 20.0f;
        glm::mat4 lightProjection = glm::ortho(-10.0f, 10.0f, -10.0f, 10.0f, near_plane, far_plane);
        
        // Calculate proper up vector for light view (perpendicular to light direction)
        glm::vec3 lightPos = -lightDir * 10.0f;
        glm::vec3 lightUp = glm::abs(lightDir.y) > 0.99f ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
        glm::mat4 lightView = glm::lookAt(lightPos, glm::vec3(0.0f), lightUp);
        glm::mat4 lightSpaceMatrix = lightProjection * lightView;
        
        // Render shadow map
        glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
        glBindFramebuffer(GL_FRAMEBUFFER, shadowMapFBO);
        glClear(GL_DEPTH_BUFFER_BIT);
        glUseProgram(shadowShader);
        
        GLuint lightSpaceLoc = glGetUniformLocation(shadowShader, "lightSpaceMatrix");
        GLuint modelLoc = glGetUniformLocation(shadowShader, "model");
        
        // Render plane to shadow map
        glUniformMatrix4fv(lightSpaceLoc, 1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(planeModel));
        glBindVertexArray(unitCubeVAO);
        glDrawElements(GL_TRIANGLES, cubeIndexCount, GL_UNSIGNED_INT, 0);
        
        // Render cube to shadow map (rotating)
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(cubeModel));
        glBindVertexArray(unitCubeVAO);
        glDrawElements(GL_TRIANGLES, cubeIndexCount, GL_UNSIGNED_INT, 0);
        glDrawElements(GL_TRIANGLES, cubeIndexCount, GL_UNSIGNED_INT, 0);
        
        // Render to offscreen framebuffer
        glViewport(0, 0, OFFSCREEN_WIDTH, OFFSCREEN_HEIGHT);
        glBindFramebuffer(GL_FRAMEBUFFER, offscreenFBO);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        glUseProgram(mainShader);
        
        // Set up view and projection
        glm::mat4 view = glm::lookAt(cameraPos, cameraTarget, cameraUp);
        glm::mat4 projection = glm::perspective(glm::radians(60.0f), (float)OFFSCREEN_WIDTH / (float)OFFSCREEN_HEIGHT, 0.1f, 100.0f);
        
        // Set uniforms
        glUniformMatrix4fv(glGetUniformLocation(mainShader, "lightSpaceMatrix"), 1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));
        glUniform3fv(glGetUniformLocation(mainShader, "lightDir"), 1, glm::value_ptr(lightDir));
        glUniform3fv(glGetUniformLocation(mainShader, "lightColor"), 1, glm::value_ptr(lightColor));
        glUniform3fv(glGetUniformLocation(mainShader, "viewPos"), 1, glm::value_ptr(cameraPos));
        glUniformMatrix4fv(glGetUniformLocation(mainShader, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(mainShader, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        
        // Bind shadow map
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, shadowMap);
        glUniform1i(glGetUniformLocation(mainShader, "shadowMap"), 0);
        
        // Render plane
        glUniformMatrix4fv(glGetUniformLocation(mainShader, "model"), 1, GL_FALSE, glm::value_ptr(planeModel));
        glUniform3fv(glGetUniformLocation(mainShader, "objectColor"), 1, glm::value_ptr(glm::vec3(0.8f, 0.8f, 0.8f)));
        glBindVertexArray(unitCubeVAO);
        glDrawElements(GL_TRIANGLES, cubeIndexCount, GL_UNSIGNED_INT, 0);
        
        // Render cube (red, rotating)
        glUniformMatrix4fv(glGetUniformLocation(mainShader, "model"), 1, GL_FALSE, glm::value_ptr(cubeModel));
        glUniform3fv(glGetUniformLocation(mainShader, "objectColor"), 1, glm::value_ptr(glm::vec3(1.0f, 0.0f, 0.0f)));
        glBindVertexArray(unitCubeVAO);
        glDrawElements(GL_TRIANGLES, cubeIndexCount, GL_UNSIGNED_INT, 0);
        
        // Render to screen
        int windowWidth, windowHeight;
        SDL_GetWindowSize(window, &windowWidth, &windowHeight);
        glViewport(0, 0, windowWidth, windowHeight);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);  // Disable culling for fullscreen quad
        
        glUseProgram(screenShader);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, offscreenColorTexture);
        glUniform1i(glGetUniformLocation(screenShader, "screenTexture"), 0);
        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        
        // Render shadow map visualization in top right corner
        glUseProgram(shadowMapVisShader);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, shadowMap);
        glUniform1i(glGetUniformLocation(shadowMapVisShader, "shadowMap"), 0);
        glBindVertexArray(shadowVisQuadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);  // Re-enable culling
        
        SDL_GL_SwapWindow(window);
    }
    
    // Cleanup
    glDeleteVertexArrays(1, &unitCubeVAO);
    glDeleteVertexArrays(1, &quadVAO);
    glDeleteVertexArrays(1, &shadowVisQuadVAO);
    glDeleteFramebuffers(1, &shadowMapFBO);
    glDeleteFramebuffers(1, &offscreenFBO);
    glDeleteTextures(1, &shadowMap);
    glDeleteTextures(1, &offscreenColorTexture);
    glDeleteTextures(1, &offscreenDepthTexture);
    glDeleteProgram(shadowShader);
    glDeleteProgram(mainShader);
    glDeleteProgram(screenShader);
    glDeleteProgram(shadowMapVisShader);
    
    SDL_GL_DestroyContext(context);
    SDL_DestroyWindow(window);
    SDL_Quit();
    
    return 0;
}

