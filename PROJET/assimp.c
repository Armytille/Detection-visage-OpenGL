/*!\file assimp.c
 *
 * \brief utilisation de GL4Dummies et Lib Assimp pour chargement de
 * scènes.
 *
 * Modification de l'exemple fourni par lib Assimp utilisant GL < 3 et
 * GLUT et upgrade avec utilisation des VAO/VBO et matrices et shaders
 * GL4dummies.
 *
 * \author Vincent Boyer et Farès Belhadj {boyer, amsi}@ai.univ-paris8.fr
 * \date February 14 2017
 */

#include <GL4D/gl4duw_SDL2.h>
#include <SDL2/SDL_image.h>

#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assert.h>

/* the global Assimp scene object */
#define _nb_max_item 30
static const struct aiScene* _scene[_nb_max_item];
static struct aiVector3D _scene_min[_nb_max_item], _scene_max[_nb_max_item], _scene_center[_nb_max_item];


#define aisgl_min(x,y) (x<y?x:y)
#define aisgl_max(x,y) (y>x?y:x)

static void get_bounding_box_for_node (const struct aiNode* nd, struct aiVector3D* min, struct aiVector3D* max, struct aiMatrix4x4* trafo,GLuint id);
static void get_bounding_box (struct aiVector3D* min, struct aiVector3D* max,GLuint id);
static void color4_to_float4(const struct aiColor4D *c, float f[4]);
static void set_float4(float f[4], float a, float b, float c, float d);
static void apply_material(const struct aiMaterial *mtl);
static void sceneMkVAOs (const struct aiScene *sc, const struct aiNode* nd, GLuint * ivao,GLuint id);
static void sceneDrawVAOs(const struct aiScene *sc, const struct aiNode* nd, GLuint * ivao,GLuint id);
static int  sceneNbMeshes(const struct aiScene *sc, const struct aiNode* nd, int subtotal);
static int  loadasset (const char* path,GLuint id);

static GLuint * _vaos[_nb_max_item], * _buffers[_nb_max_item], * _counts[_nb_max_item], * _textures[_nb_max_item], _nbMeshes[_nb_max_item], _nbTextures[_nb_max_item];


/*!\brief modification du Assimp.c avec l'ajout d'un id afin de charger plusieurs objets différents en même temps
 *
 * \param le nom du fichier objet
 * + l'id de l'objet (un GLuint qui permet de différencier nos objets entre eux)
 */
void assimpInit(const char * filename, GLuint id) {
  int i;
  GLuint ivao = 0;
  _scene[id] = NULL;
  _vaos[id] = NULL;
  _buffers[id] = NULL;
  _counts[id] = NULL;
  _textures[id] = NULL;
  _nbMeshes[id] = 0;
  _nbTextures[id] = 0;

  struct aiLogStream stream;
  /* get a handle to the predefined STDOUT log stream and attach
     it to the logging system. It remains active for all further
     calls to aiImportFile(Ex) and aiApplyPostProcessing. */
  stream = aiGetPredefinedLogStream(aiDefaultLogStream_STDOUT, NULL);
  aiAttachLogStream(&stream);
  /* ... same procedure, but this stream now writes the
     log messages to assimp_log.txt */
  stream = aiGetPredefinedLogStream(aiDefaultLogStream_FILE,"assimp_log.txt");
  aiAttachLogStream(&stream);
  /* the model name can be specified on the command line. If none
     is specified, we try to locate one of the more expressive test 
     models from the repository (/models-nonbsd may be missing in 
     some distributions so we need a fallback from /models!). */
  if(loadasset(filename,id) != 0) {
    fprintf(stderr, "Erreur lors du chargement du fichier %s\n", filename);
    exit(3);
  } 
  /* XXX docs say all polygons are emitted CCW, but tests show that some aren't. */
  if(getenv("MODEL_IS_BROKEN"))  
    glFrontFace(GL_CW);


  _textures[id] = malloc((_nbTextures[id] = _scene[id]->mNumMaterials) * sizeof *_textures[id]);
  assert(_textures[id]);
  
  glGenTextures(_nbTextures[id], _textures[id]);

  for (i = 0; i < _scene[id]->mNumMaterials ; i++) {
    const struct aiMaterial* pMaterial = _scene[id]->mMaterials[i];
    if (aiGetMaterialTextureCount(pMaterial, aiTextureType_DIFFUSE) > 0) {
      struct aiString tfname;
      char * dir = pathOf(filename), buf[BUFSIZ];
      if (aiGetMaterialTexture(pMaterial, aiTextureType_DIFFUSE, 0, &tfname, NULL, NULL, NULL, NULL, NULL, NULL) == AI_SUCCESS) {
	SDL_Surface * t;
	snprintf(buf, sizeof buf, "%s/%s", dir, tfname.data);

	if(!(t = IMG_Load(buf))) { 
	  fprintf(stderr, "Probleme de chargement de textures %s\n", buf); 
	  fprintf(stderr, "\tNouvel essai avec %s\n", tfname.data);
	  if(!(t = IMG_Load(tfname.data))) { fprintf(stderr, "Probleme de chargement de textures %s\n", tfname.data); continue; }
	}
	glBindTexture(GL_TEXTURE_2D, _textures[id][i]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT/* GL_CLAMP_TO_EDGE */);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT/* GL_CLAMP_TO_EDGE */);
#ifdef __APPLE__
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, t->w, t->h, 0, t->format->BytesPerPixel == 3 ? GL_BGR : GL_BGRA, GL_UNSIGNED_BYTE, t->pixels);
#else
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, t->w, t->h, 0, t->format->BytesPerPixel == 3 ? GL_RGB : GL_RGBA, GL_UNSIGNED_BYTE, t->pixels);
#endif
	SDL_FreeSurface(t);
      }
    }
  }

  _nbMeshes[id]= sceneNbMeshes(_scene[id], _scene[id]->mRootNode, 0);
  _vaos[id] = malloc(_nbMeshes[id] * sizeof *_vaos[id]);
  assert(_vaos[id]);
  glGenVertexArrays(_nbMeshes[id], _vaos[id]);
  _buffers[id] = malloc(2 * _nbMeshes[id] * sizeof *_buffers[id]);
  assert(_buffers[id]);
  glGenBuffers(2 * _nbMeshes[id], _buffers[id]);
  _counts[id] = calloc(_nbMeshes[id], sizeof *_counts[id]);
  assert(_counts[id]);
  sceneMkVAOs(_scene[id], _scene[id]->mRootNode, &ivao,id);
}

void assimpDrawScene(GLuint id) {
  GLfloat tmp;
  GLuint ivao = 0;
  tmp = _scene_max[id].x - _scene_min[id].x;
  tmp = aisgl_max(_scene_max[id].y - _scene_min[id].y, tmp);
  tmp = aisgl_max(_scene_max[id].z - _scene_min[id].z, tmp);
  tmp = 1.0f / tmp;
  gl4duScalef(tmp, tmp, tmp);
  gl4duTranslatef( -_scene_center[id].x, -_scene_center[id].y, -_scene_center[id].z);
  sceneDrawVAOs(_scene[id], _scene[id]->mRootNode, &ivao,id);
}

void assimpQuit(void) {
  /* cleanup - calling 'aiReleaseImport' is important, as the library 
     keeps internal resources until the scene is freed again. Not 
     doing so can cause severe resource leaking. */
  int id;
  for(id=0;id<_nb_max_item;id++)
    aiReleaseImport(_scene[id]);
  /* We added a log stream to the library, it's our job to disable it
     again. This will definitely release the last resources allocated
     by Assimp.*/
  aiDetachAllLogStreams();
  for(id=0;id<_nb_max_item;id++){
    if(_counts[id]) {
      free(_counts[id]);
      _counts[id] = NULL;
    }
    if(_textures[id]) {
      glDeleteTextures(_nbTextures[id], _textures[id]);
      free(_textures[id]);
      _textures[id] = NULL;
    }
    if(_vaos[id]) {
      glDeleteVertexArrays(_nbMeshes[id], _vaos[id]);
      free(_vaos[id]);
      _vaos[id] = NULL;
    }
    if(_buffers[id]) {
      glDeleteBuffers(2 * _nbMeshes[id], _buffers[id]);
      free(_buffers[id]);
      _buffers[id] = NULL;
    }
  }

}

static void get_bounding_box_for_node(const struct aiNode* nd, struct aiVector3D* min, struct aiVector3D* max, struct aiMatrix4x4* trafo, GLuint id) {
  struct aiMatrix4x4 prev;
  unsigned int n = 0, t;
  prev = *trafo;
  aiMultiplyMatrix4(trafo,&nd->mTransformation);
  for (; n < nd->mNumMeshes; ++n) {
    const struct aiMesh* mesh = _scene[id]->mMeshes[nd->mMeshes[n]];
    for (t = 0; t < mesh->mNumVertices; ++t) {
      struct aiVector3D tmp = mesh->mVertices[t];
      aiTransformVecByMatrix4(&tmp,trafo);
      min->x = aisgl_min(min->x,tmp.x);
      min->y = aisgl_min(min->y,tmp.y);
      min->z = aisgl_min(min->z,tmp.z);
      max->x = aisgl_max(max->x,tmp.x);
      max->y = aisgl_max(max->y,tmp.y);
      max->z = aisgl_max(max->z,tmp.z);
    }
  }
  for (n = 0; n < nd->mNumChildren; ++n) {
    get_bounding_box_for_node(nd->mChildren[n],min,max,trafo,id);
  }
  *trafo = prev;
}

static void get_bounding_box (struct aiVector3D* min, struct aiVector3D* max,GLuint id) {
  struct aiMatrix4x4 trafo;
  aiIdentityMatrix4(&trafo);
  min->x = min->y = min->z =  1e10f;
  max->x = max->y = max->z = -1e10f;
  get_bounding_box_for_node(_scene[id]->mRootNode,min,max,&trafo,id);
}

static void color4_to_float4(const struct aiColor4D *c, float f[4]) {
  f[0] = c->r; f[1] = c->g; f[2] = c->b; f[3] = c->a;
}

static void set_float4(float f[4], float a, float b, float c, float d) {
  f[0] = a; f[1] = b; f[2] = c; f[3] = d;
}

static void apply_material(const struct aiMaterial *mtl) {
  float c[4];
  unsigned int max;
  float shininess, strength;
  struct aiColor4D diffuse, specular, ambient, emission;
  GLint id;
  glGetIntegerv(GL_CURRENT_PROGRAM, &id);
  
  set_float4(c, 0.8f, 0.8f, 0.8f, 1.0f);
  if (AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_DIFFUSE, &diffuse)){
    color4_to_float4(&diffuse, c);
  }
  glUniform4fv(glGetUniformLocation(id, "diffuse_color"), 1, c);

  set_float4(c, 0.0f, 0.0f, 0.0f, 1.0f);
  if (AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_SPECULAR, &specular)){
    color4_to_float4(&specular, c);
  }
  glUniform4fv(glGetUniformLocation(id, "specular_color"), 1, c);

  set_float4(c, 0.2f, 0.2f, 0.2f, 1.0f);
  if (AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_AMBIENT, &ambient)){
    color4_to_float4(&ambient, c);
  }
  glUniform4fv(glGetUniformLocation(id, "ambient_color"), 1, c);

  set_float4(c, 0.0f, 0.0f, 0.0f, 1.0f);
  if (AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_EMISSIVE, &emission)){
    color4_to_float4(&emission, c);
  }
  glUniform4fv(glGetUniformLocation(id, "emission_color"), 1, c);

  max = 1;
  if(aiGetMaterialFloatArray(mtl, AI_MATKEY_SHININESS, &shininess, &max) == AI_SUCCESS) {
    max = 1;
    if(aiGetMaterialFloatArray(mtl, AI_MATKEY_SHININESS_STRENGTH, &strength, &max) == AI_SUCCESS)
      glUniform1f(glGetUniformLocation(id, "shininess"), shininess * strength);
    else
	glUniform1f(glGetUniformLocation(id, "shininess"), shininess);
  } else {
    shininess = 0.0;
    glUniform1f(glGetUniformLocation(id, "shininess"), shininess);
  }
}

static void sceneMkVAOs(const struct aiScene *sc, const struct aiNode* nd, GLuint * ivao, GLuint id) {
  int i, j, comp;
  unsigned int n = 0;
  static int temp = 0;

  temp++;

  for (; n < nd->mNumMeshes; ++n) {
    GLfloat * vertices = NULL;
    GLuint  * indices  = NULL;
    const struct aiMesh* mesh = sc->mMeshes[nd->mMeshes[n]];
    comp  = mesh->mVertices ? 3 : 0;
    comp += mesh->mNormals ? 3 : 0;
    comp += mesh->mTextureCoords[0] ? 2 : 0;
    if(!comp) continue;

    glBindVertexArray(_vaos[id][*ivao]);
    glBindBuffer(GL_ARRAY_BUFFER, _buffers[id][2 * (*ivao)]);

    vertices = malloc(comp * mesh->mNumVertices * sizeof *vertices);
    assert(vertices);
    i = 0;
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    if(mesh->mVertices) {
      glEnableVertexAttribArray(0);
      glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (const void *)(i * sizeof *vertices));
      for(j = 0; j < mesh->mNumVertices; ++j) {
	vertices[i++] = mesh->mVertices[j].x;
	vertices[i++] = mesh->mVertices[j].y;
	vertices[i++] = mesh->mVertices[j].z;
      }      
    }
    if(mesh->mNormals) {
      glEnableVertexAttribArray(1);
      glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (const void *)(i * sizeof *vertices));
      for(j = 0; j < mesh->mNumVertices; ++j) {
	vertices[i++] = mesh->mNormals[j].x;
	vertices[i++] = mesh->mNormals[j].y;
	vertices[i++] = mesh->mNormals[j].z;
      }      
    }
    if(mesh->mTextureCoords[0]) {
      glEnableVertexAttribArray(2);
      glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, (const void *)(i * sizeof *vertices));
      for(j = 0; j < mesh->mNumVertices; ++j) {
	vertices[i++] = mesh->mTextureCoords[0][j].x;
	vertices[i++] = mesh->mTextureCoords[0][j].y;
      }      
    }
    glBufferData(GL_ARRAY_BUFFER, (i * sizeof *vertices), vertices, GL_STATIC_DRAW);      
    free(vertices);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _buffers[id][2 * (*ivao) + 1]);
    if(mesh->mFaces) {
      indices = malloc(3 * mesh->mNumFaces * sizeof *indices);
      assert(indices);
      for(i = 0, j = 0; j < mesh->mNumFaces; ++j) {
	assert(mesh->mFaces[j].mNumIndices < 4);
	if(mesh->mFaces[j].mNumIndices != 3) continue;
	indices[i++] = mesh->mFaces[j].mIndices[0];
	indices[i++] = mesh->mFaces[j].mIndices[1];
	indices[i++] = mesh->mFaces[j].mIndices[2];
      }
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, i * sizeof *indices, indices, GL_STATIC_DRAW);
      _counts[id][*ivao] = i;
      free(indices);
    }
    glBindVertexArray(0);
    (*ivao)++;
  }
  for (n = 0; n < nd->mNumChildren; ++n) {
    sceneMkVAOs(sc, nd->mChildren[n], ivao,id);
  }
}


static void sceneDrawVAOs(const struct aiScene *sc, const struct aiNode* nd, GLuint * ivao,GLuint i) {
  unsigned int n = 0;
  struct aiMatrix4x4 m = nd->mTransformation;
  GLint id;

  glGetIntegerv(GL_CURRENT_PROGRAM, &id);
  /* By VB Inutile de transposer la matrice, gl4dummies fonctionne avec des transpose de GL. */
  /* aiTransposeMatrix4(&m); */
  gl4duPushMatrix();
  gl4duMultMatrixf((GLfloat*)&m);
  gl4duSendMatrices();

  for (; n < nd->mNumMeshes; ++n) {
    const struct aiMesh* mesh = sc->mMeshes[nd->mMeshes[n]];
    if(_counts[i][*ivao]) {
      glBindVertexArray(_vaos[i][*ivao]);
      apply_material(sc->mMaterials[mesh->mMaterialIndex]);
      if (aiGetMaterialTextureCount(sc->mMaterials[mesh->mMaterialIndex], aiTextureType_DIFFUSE) > 0) {
	glBindTexture(GL_TEXTURE_2D, _textures[i][mesh->mMaterialIndex]);
	glUniform1i(glGetUniformLocation(id, "hasTexture"), 1);
	glUniform1i(glGetUniformLocation(id, "myTexture"), 0);
      } else {
	glUniform1i(glGetUniformLocation(id, "hasTexture"), 0);
      }
      glDrawElements(GL_TRIANGLES, _counts[i][*ivao], GL_UNSIGNED_INT, 0);
      glBindVertexArray(0);
      glBindTexture(GL_TEXTURE_2D, 0);
    }
    (*ivao)++;
  }  
  for (n = 0; n < nd->mNumChildren; ++n) {
    sceneDrawVAOs(sc, nd->mChildren[n], ivao,i);
  }
  gl4duPopMatrix(); 
}

static int sceneNbMeshes(const struct aiScene *sc, const struct aiNode* nd, int subtotal) {
  int n = 0;
  subtotal += nd->mNumMeshes;
  for(n = 0; n < nd->mNumChildren; ++n)
    subtotal += sceneNbMeshes(sc, nd->mChildren[n], 0);
  return subtotal;
}

static int loadasset (const char* path,GLuint id) {
  /* we are taking one of the postprocessing presets to avoid
     spelling out 20+ single postprocessing flags here. */
  /* struct aiString str; */
  /* aiGetExtensionList(&str); */
  /* fprintf(stderr, "EXT %s\n", str.data); */
  _scene[id] = aiImportFile(path, 
		       aiProcessPreset_TargetRealtime_MaxQuality |
		       aiProcess_CalcTangentSpace       |
		       aiProcess_Triangulate            |
		       aiProcess_JoinIdenticalVertices  |
		       aiProcess_SortByPType);
  if (_scene[id]) {
    get_bounding_box(&_scene_min[id],&_scene_max[id],id);
    _scene_center[id].x = (_scene_min[id].x + _scene_max[id].x) / 2.0f;
    _scene_center[id].y = (_scene_min[id].y + _scene_max[id].y) / 2.0f;
    _scene_center[id].z = (_scene_min[id].z + _scene_max[id].z) / 2.0f;
    return 0;
  }
  return 1;
}
