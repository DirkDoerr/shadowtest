This is primarally a test for vibe coding with Cursor (auto AI selection).

Initial prompt:
I want a new project that renders a 3d scene using shadows and simple phong shading in OpenGL

- it uses cmake to configure
- c++ 20
- the folder already contains code to loading OpenGL
- it should use SDL 3
- SingleFile shadowtest.cpp
- Shader should be embedded
- renders a 10x10 plane with a 4 unit thickness at origin (-4 below)
- a 1x1x1 red unit cube rotating around origin
- it should render into a offscreen framebuffer and then render the result into the main window
- shadow map 2048x2048
- rendering fullHD offscreen


Make a plan and the we review and get started


Result:
Almost one-shotted. 
Initial problems with winding order of vertices

Then we had issues with lighting and backface culling. But after about 2 hours (maybe less) it was running correctly. 


Dirk Dörr, Dezember 2025