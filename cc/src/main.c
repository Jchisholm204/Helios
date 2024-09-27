#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <memory.h>
#include <sys/stat.h>
#include <stdbool.h>

// Maximum length of a typename
#define MAX_TYPE 50

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
    struct data_type * vars;
    size_t n_vars;
    struct context *internal;
    struct context *external;
    size_t n_internal;
};

// Object file (contains contexts)
struct object {
    FILE *pFile;
    char * pBuffer;
    size_t n_buffer;
    size_t len_obj;
    struct context * context;
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
    pObj->pBuffer = NULL;
    pObj->n_buffer = 0;

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
    obj->pBuffer = malloc(obj->len_obj);

    char c = fgetc(obj->pFile);
    char cl = 0;
    int i = 0;
    // Scan the file
    while(c != EOF){
        // Ignore new lines and extra spaces
        if(c != '\n' && !(c == ' ' && cl == ' ')){
            obj->pBuffer[i++] = c;
        }
        cl = c;
        c = fgetc(obj->pFile);
    }
    obj->pBuffer[i] = '\0';
    obj->pBuffer = realloc(obj->pBuffer, i+1);
    obj->n_buffer = i;
}

void obj_parse(struct object * pObj){
    if(!pObj){
        fprintf(stderr, "parse recvd null object\n");
        exit(1);
    }
    bool outsideContext = false;
    bool nestedContext = false;
    struct context * currentContext = NULL;
    
    for(int i = 0; i < pObj->n_buffer; i++){
        char expression[MAX_TYPE] = {'\0'};
        // Get the next expression
        for(int j = i; j < MAX_TYPE && j < pObj->n_buffer; j++){
            expression[j-i] = pObj->pBuffer[j];
            char c = pObj->pBuffer[j];
            if(!((c > 'A' && c < 'Z') || (c > '0' && c < '9') || (c > 'a' && c < 'z')))
                break;
        }
        // Figure out the context stuff
    }
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
    fclose(obj->pFile);
    printf("%s\n", obj->pBuffer);
    free(obj->pBuffer);
    free(obj);
    return 0;
}
