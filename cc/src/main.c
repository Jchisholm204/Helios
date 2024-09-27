#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <memory.h>
#include <sys/stat.h>
#include <stdbool.h>

// Maximum length of a typename
#define MAX_TYPE 500
#define MAX_BUF  100

// Struct to hold a variable
struct data_type {
    char *name;
    enum eType {eCHAR, eU32, eVOID} type;
    union {
        uint32_t u32int;
        char     u8int;
    };
    struct context *context;
};

struct operation_type{
    enum eOp {eADD, eSUB} type;
    struct data_type *var1, *var2;
};

// Struct to hold a context {this is a context {nested context}}
struct context {
    int depth;
    char *name;
    struct data_type ret;
    // List of all vars created in this context
    struct data_type * vars;
    size_t n_vars;
    struct data_type * params;
    size_t n_params;
    // List of all operations in this context
    struct operation_type * operations;
    size_t n_operations;
    struct context *internal;
    struct context *external;
    size_t n_internal;
};

// Object file (contains contexts)
struct object {
    FILE *pFile;
    char **objects;
    size_t n_objects;
    size_t len_obj;
    struct context * context;
    struct context base_context;
    size_t n_context;
};

// Open an object for parsing
struct object * obj_open(char * fname){
    // Allocate memory for the object file
    struct object * pObj = malloc(sizeof(struct object));
    if(!pObj){
        fprintf(stderr, "Failed to allocate memory for object\n");
        exit(1);
    }

    // Open the raw object file as a binary object
    pObj->pFile = fopen(fname, "rb");

    if(!pObj->pFile){
        fprintf(stderr, "Could not open file %s\n", fname);
        exit(1);
    }

    // Get the size of the file in bytes
    struct stat fstat;
    stat(fname, &fstat);
    pObj->len_obj = fstat.st_size;

    // The file has no contexts to start with
    pObj->context = NULL;
    pObj->n_context = 0;
    pObj->objects = NULL;
    pObj->n_objects = 0;

    return pObj;
}

// Scan an object before parsing
void obj_scan(struct object * obj){
    // Handle null object
    if(!obj){
        fprintf(stderr, "scan recvd null object\n");
        exit(1);
    }

    // Allocate a char buffer to hold the base file contents
    obj->objects = (char**)malloc(obj->len_obj*sizeof(char*));

    char * fbuf = (char*)malloc(obj->len_obj);

    char c = fgetc(obj->pFile);
    char cl = 0;
    int n_buf = 0;
    // Scan the file
    while(c != EOF){
        // Ignore new lines and extra spaces
        if(c != '\n' && !(c == ' ' && cl == ' ')){
            fbuf[n_buf++] = c;
        }
        cl = c;
        c = fgetc(obj->pFile);
    }
    fbuf[n_buf] = '\0';

    obj->n_objects = 0;

    for(int i = 0; i < n_buf;){
        char expression[MAX_TYPE] = {'\0'};
        // Get the next expression
        int j = 0;
        bool string = false;
        for(; j < MAX_TYPE; j++){
            expression[j] = fbuf[j+i];
            char c = expression[j];
            if(string && c == '"'){
                if(j == 0)
                    j = 1;
                j++;
                break;
            }
            if(c == '"')
                string = true;
            // printf("%c ", c);
            if(!string && !((c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z'))){
                if(j == 0)
                    j = 1;
                // if(c == ';')
                //     j++;
                break;
            }
        }
        expression[j] = '\0';
        i += j;
        // printf("%s (%d)\n", expression, i);
        if(expression[0] != ' '){
            obj->objects[obj->n_objects] = malloc(strlen(expression));
            strcpy(obj->objects[obj->n_objects++], expression);
        }
    }

    // free(fbuf);

}

void var_parse(struct object * pObj, struct context * pCtx, int i, char delim){

}


void context_parse(struct object * pObj, struct context * pCtx, size_t *start);

void func_parse(struct object * pObj, struct context * pCtx, size_t *start){
    printf("Parsing Function @ %d\n", *start);
    if(pCtx->n_internal == 0)
        pCtx->internal = malloc(MAX_BUF*sizeof(struct context));
    struct context *ctx = &pCtx->internal[pCtx->n_internal++];
    ctx->n_internal = 0;
    ctx->internal = NULL;
    ctx->depth = pCtx->depth + 1;
    ctx->name = pObj->objects[*start+1];
    ctx->ret.name = ctx->name;
    ctx->ret.context = ctx;
    ctx->ret.type = eU32;
    int j = *start;
    ctx->params = malloc(MAX_BUF*sizeof(struct data_type));
    ctx->n_params = 0;
    for(; j < pObj->n_objects; j++){
        char *expr = pObj->objects[j];
        printf("fexpression %d: %s\n", j, expr);
        // printf("Parsing func portion %s\n", expr);
        if(expr[0] ==  ')'){
            break;
        }
        if(expr[0] ==  ','){
            continue;
        }

        if(!strcmp(expr, "int")){
            struct data_type *params =  &ctx->params[ctx->n_params++];
            params->name = pObj->objects[j+1];
            params->context = ctx;
            params->type = eU32;
        }
        else if(!strcmp(expr, "char")){
            struct data_type *params =  &ctx->params[ctx->n_params++];
            params->name = pObj->objects[j+1];
            params->context = ctx;
            params->type = eCHAR;
        }

    }
    printf("Parsed function with %d params\n", ctx->n_params);
    for(int i = 0; i < ctx->n_params; i++){
        printf("param %d: %d %s\n", i, ctx->params[i].type, ctx->params[i].name);
    }
    *start = j;
    // context_parse(pObj, ctx, start);
}

void context_parse(struct object * pObj, struct context * pCtx, size_t *start){
    for(; *start< pObj->n_objects; (*start)++){
        char * expr = pObj->objects[*start];
        printf("expression %d: %s\n", *start, expr);
        // If I'm looking at a new integer
        if(!strcmp(expr, "int")){
            // This must be a new function
            if(pObj->objects[*start+2][0] == '('){
                *start += 3;
                func_parse(pObj, pCtx, start);
                printf("finished parsing function @ %d\n", *start);
            }
        }
        else if(!strcmp(expr, "void")){
        }
    }
}

void obj_parse(struct object * pObj){
    if(!pObj){
        fprintf(stderr, "parse recvd null object\n");
        exit(1);
    }
    bool outsideContext = true;
    struct context * currentContext = NULL;

    // First we need a base context to put things in
    currentContext = &pObj->base_context;
    currentContext->name = "base";
    currentContext->depth = 0;
    currentContext->ret = (struct data_type){"base", eVOID, 0, currentContext};
    currentContext->n_internal = 0;
    
    for(int i = 0; i < pObj->n_objects; i++){
        printf("%d: %s\n", i, pObj->objects[i]);
    }

    // Start by parsing the base context
    size_t start = 0;
    context_parse(pObj, currentContext, &start);
}

int main(int argc, char **argv){
    printf("argc: %d\n", argc);
    printf("argv: %s\n", argv[0]);
    if(argc < 2){
        fprintf(stderr, "At least one argument is required.\n");
        exit(1);
    }
    struct object * obj = obj_open(argv[1]);
    obj_scan(obj);
    obj_parse(obj);
    fclose(obj->pFile);
    // printf("%s\n", obj->pBuffer);
    // free(obj->pBuffer);
    free(obj);
    return 0;
}
