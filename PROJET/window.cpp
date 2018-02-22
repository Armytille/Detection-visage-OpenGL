#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio/videoio_c.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <GL4D/gl4du.h>
#include <GL4D/gl4dg.h>
#include <GL4D/gl4duw_SDL2.h>
#include <SDL2/SDL_image.h>
#include "assimp.h"

using namespace cv;
using namespace std;

static CascadeClassifier * face_cc;
static CascadeClassifier * eye_cc;
static CascadeClassifier * nose_cc;

static Mat cameraFrame;

/*!\brief dimensions de la fenêtre */
static int _windowWidth = 800, _windowHeight = 600;
/*!\brief pointeur vers la (future) fenêtre SDL */
static SDL_Window * _win = NULL;
/*!\brief pointeur vers le (futur) contexte OpenGL */
static SDL_GLContext _oglContext = NULL;
/*!\brief identifiant du (futur) vertex array object */
static GLuint _vao = 0;
/*!\brief identifiant du (futur) buffer de data */
static GLuint _buffer = 0;
/*!\brief identifiants des (futurs) GLSL programs */
static GLuint _pId = 0, _obj_pId = 0;
/*!\brief identifiant de la texture chargée */
static GLuint _tId = 0;
/*!\brief device de capture vidéo */
static VideoCapture * camera = NULL;

/* fonctions locales, statiques */
static SDL_Window * initWindow(int w, int h, SDL_GLContext * poglContext);
static void initGL(SDL_Window * win);
static void initData(void);
static void resizeGL(SDL_Window * win);
static void loop(SDL_Window * win);
static void draw(void);
static void quit(void);

static SDL_Window * initWindow(int w, int h, SDL_GLContext * poglContext) {
  SDL_Window * win = NULL;
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
  if((win = SDL_CreateWindow("cam", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 
            w, h, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | 
            SDL_WINDOW_SHOWN)) == NULL )
    return NULL;
  if((*poglContext = SDL_GL_CreateContext(win)) == NULL ) {
    SDL_DestroyWindow(win);
    return NULL;
  }
  fprintf(stderr, "Version d'OpenGL : %s\n", glGetString(GL_VERSION));
  fprintf(stderr, "Version de shaders supportes : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));  
  return win;
}

/*!\brief Cette fonction initialise les paramètres OpenGL.
 *
 * \param win le pointeur vers la fenêtre SDL pour laquelle nous avons
 * attaché le contexte OpenGL.
 */
static void initGL(SDL_Window * win) {
  /*initAssimp*/
  assimpInit("glasses/Glasses.obj", 0);
  assimpInit("mustache/Mustache.obj", 1);
  /*initGL*/
  glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
  gl4duGenMatrix(GL_FLOAT, "projectionMatrix");
  gl4duGenMatrix(GL_FLOAT, "modelviewMatrix");
  gl4duBindMatrix("modelviewMatrix");
  gl4duLoadIdentityf();
  /* placer les objets en -10, soit bien après le plan near (qui est à -2 voir resizeGL) */
  gl4duTranslatef(0, 0, -10);
  resizeGL(win);
}

/*!\brief Cette fonction paramétrela vue (viewPort) OpenGL en fonction
 * des dimensions de la fenêtre SDL pointée par \a win.
 *
 * \param win le pointeur vers la fenêtre SDL pour laquelle nous avons
 * attaché le contexte OpenGL.
 */
static void resizeGL(SDL_Window * win) {
  int w, h;
  SDL_GetWindowSize(win, &w, &h);
  glViewport(0, 0, w, h);
  gl4duBindMatrix("projectionMatrix");
  gl4duLoadIdentityf();
  gl4duFrustumf(-1.0f, 1.0f, -h / (GLfloat)w, h / (GLfloat)w, 2.0f, 1000.0f);
}

static void initData(void) {
  GLfloat data[] = {
    /* 4 coordonnées de sommets */
    -1.f, -1.f, 0.f, 1.f, -1.f, 0.f,
    -1.f,  1.f, 0.f, 1.f,  1.f, 0.f,
    /* 2 coordonnées de texture par sommet */
    1.0f, 1.0f, 0.0f, 1.0f, 
    1.0f, 0.0f, 0.0f, 0.0f
  };
  glGenVertexArrays(1, &_vao);
  glBindVertexArray(_vao);

  glGenBuffers(1, &_buffer);
  glBindBuffer(GL_ARRAY_BUFFER, _buffer);
  glBufferData(GL_ARRAY_BUFFER, sizeof data, data, GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1); 
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (const void *)0);  
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (const void *)((4 * 3) * sizeof *data));
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  glGenTextures(1, &_tId);
  glBindTexture(GL_TEXTURE_2D, _tId);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  
  face_cc = new CascadeClassifier("haarcascade_frontalface_default.xml");
  eye_cc = new CascadeClassifier("haarcascade_eye.xml");
  nose_cc = new CascadeClassifier("Nariz.xml");


  if(face_cc == NULL || eye_cc == NULL || nose_cc == NULL){
    cout << "impossible d'ouvrir les .xml";
    return;
  }
  camera = new VideoCapture(0);
  if(!camera || !camera->isOpened()) {
    delete camera;
    camera = new VideoCapture(CV_CAP_ANY);
  }
  camera->set(CV_CAP_PROP_FRAME_WIDTH,  (int)_windowWidth);
  camera->set(CV_CAP_PROP_FRAME_HEIGHT, (int)_windowHeight);
}

/*!\brief Boucle infinie principale : gère les évènements, dessine,
 * imprime le FPS et swap les buffers.
 *
 * \param win le pointeur vers la fenêtre SDL pour laquelle nous avons
 * attaché le contexte OpenGL.
 */
static void loop(SDL_Window * win) {
  SDL_Event event;
  while (event.type != SDL_QUIT) {
        draw();
        //gl4duPrintFPS(stderr);
        SDL_GL_SwapWindow(win);
        gl4duUpdateShaders();
        if((waitKey(10)& 0xFF)== 27)
          break;
        SDL_PollEvent(&event);
  }
}

/*!\brief traduit les coordonnees renvoyées par vector<Rect>,
 * en coordonnees exploitables par le contexte OpenGL.
 *
 * \param pointeur ip: tableau contenant le x,y et z que l'on rempli dans la fonction
 * valeur x de Rect
 * valeur y de Rect
 */
static void translate_coord(GLfloat * ip, int xm, int ym) {
    GLfloat m[16], mvmatrix[16], projmatrix[16],* gl4dm;
    GLfloat p[] = { -(180.0f * xm / (GLfloat)_windowWidth - 1.0f),
        -(200.0f * ym / (GLfloat)_windowHeight - 1.0f), 
        1.0f, 1.0 }, mcoords[4] = {0, 0, 0, 1}, mscr[4];
    gl4duBindMatrix("projectionMatrix");
    gl4dm = (GLfloat*)gl4duGetMatrixData();
    memcpy(projmatrix, gl4dm, sizeof projmatrix);
    gl4duBindMatrix("modelViewMatrix");
    gl4dm = (GLfloat*)gl4duGetMatrixData();
    memcpy(mvmatrix, gl4dm, sizeof mvmatrix);
    MMAT4XMAT4(m, projmatrix, mvmatrix);
    MMAT4XVEC4(mscr, m, mcoords);
    MVEC4WEIGHT(mscr);
    p[2] = mscr[2];
    MMAT4INVERSE(m);
    MMAT4XVEC4(ip, m, p);
    MVEC4WEIGHT(ip);
}

/*!\brief fonction qui dessine un object assimp
 *
 * \param x, y, z, theta et l'id de notre object
 */
void assimpObjet(GLfloat x, GLfloat y, GLfloat z, GLfloat theta, GLuint id) {
  gl4duPushMatrix(); {
    gl4duTranslatef(x, y, z);
    gl4duScalef(50, 50, 5);
    gl4duRotatef(theta, 0, 1, 0);
    gl4duSendMatrices();
    assimpDrawScene(id);
  } gl4duPopMatrix();
}

/*!\brief dessine dans le contexte OpenGL actif. */
static void draw(void) {
  const GLfloat blanc[] = {1.0f, 1.0f, 1.0f, 1.0f};
  *camera >> cameraFrame;
  vector<Rect> faces;
 face_cc->detectMultiScale(cameraFrame, faces, 1.2, 5);
 
  glBindTexture(GL_TEXTURE_2D, _tId);
 
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, cameraFrame.cols, cameraFrame.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, cameraFrame.data);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glUseProgram(_pId);
  glEnable(GL_DEPTH_TEST);
  glUniform1i(glGetUniformLocation(_pId, "myTexture"), 0);
  glUniform1f(glGetUniformLocation(_pId, "width"), _windowWidth);
  glUniform1f(glGetUniformLocation(_pId, "height"), _windowHeight);
  /* streaming au fond */
  gl4duBindMatrix("modelviewMatrix");
  gl4duPushMatrix(); /* sauver modelview */
  gl4duLoadIdentityf();
  gl4duTranslatef(0, 0, 0.9999f);
  gl4duBindMatrix("projectionMatrix");
  gl4duPushMatrix(); /* sauver projection */
  gl4duLoadIdentityf();
  gl4duSendMatrices(); /* envoyer les matrices */
  glUniform4fv(glGetUniformLocation(_pId, "couleur"), 1, blanc); /* envoyer une couleur */
  glBindVertexArray(_vao);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4); /* dessiner le streaming (ortho et au fond) */
  gl4duPopMatrix(); /* restaurer projection */
  gl4duBindMatrix("modelviewMatrix");
  gl4duPopMatrix(); /* restaurer modelview */

  GLfloat ip[4];
  for (vector<Rect>::iterator fc = faces.begin(); fc != faces.end(); ++fc) { //Détecte chaque visages
    //cout << ip[2] << '\n';
    translate_coord(ip, (int)((*fc).tl()).x, (int)((*fc).tl()).y); 
    assimpObjet(ip[0]+70, ip[1]+35, (GLfloat)(((*fc).width*(*fc).height)/1000)-250, -10, 0);
    Mat cameraFrame_roi = cameraFrame(*fc);
    vector<Rect> noses;      
    nose_cc->detectMultiScale(cameraFrame_roi, noses, 1.3, 10);
    for(vector<Rect>::iterator nc = noses.begin(); nc != noses.end(); ++nc){
      translate_coord(ip, (int)((*nc).tl()).x, (int)((*nc).tl()).y); 
      assimpObjet(ip[0]+22, ip[1]+15, (GLfloat)(((*nc).width*(*nc).height)/1000)-150, -10, 1);
    }
     
  }

}

/*!\brief appelée au moment de sortir du programme (atexit), libère les éléments utilisés */
static void quit(void) {
  //Le destructeur par défaut s'en charge automatiquement.
  /*if(camera) {
    delete camera;
    camera = NULL;
  } */ 

  if(_vao)
    glDeleteVertexArrays(1, &_vao);
  if(_buffer)
    glDeleteBuffers(1, &_buffer);
  if(_tId)
    glDeleteTextures(1, &_tId);
  if(_oglContext)
    SDL_GL_DeleteContext(_oglContext);
  if(_win)
    SDL_DestroyWindow(_win);
  gl4duClean(GL4DU_ALL);
  assimpQuit();
}

int main(int argc, char ** argv) {
  if(SDL_Init(SDL_INIT_VIDEO) < 0) {
    fprintf(stderr, "Erreur lors de l'initialisation de SDL :  %s", SDL_GetError());
    return -1;
  }
  SDL_GL_SetSwapInterval(1);
  //atexit(SDL_Quit);
  if((_win = initWindow(_windowWidth, _windowHeight, &_oglContext))) {
    atexit(quit);
    gl4duInit(argc, argv);
    initGL(_win);
    _pId = gl4duCreateProgram("<vs>shaders/basic.vs", "<fs>shaders/basic.fs", NULL);
    initData();
    loop(_win);
  }

  return 0;
}
