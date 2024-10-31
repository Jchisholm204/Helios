#include "main.h"
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <err.h>

struct program {
    uint32_t *memory;
    size_t memory_size;
    size_t n_instructions;
};

struct program * input_hndl(char *fname){
    FILE *pFile = fopen(fname, "rb");
    if(!pFile) return NULL;
    struct program * pProg = malloc(sizeof(struct program));
    if(!pProg){
        fclose(pFile);
        return NULL;
    }
    fseek(pFile, 0, SEEK_END);
    pProg->memory_size = ftell(pFile);
    fseek(pFile, 0, SEEK_SET);
    if((pProg->memory_size % 4) != 0){
        fprintf(stderr, "program size not divisible by 4\n");
        free(pProg);
        fclose(pFile);
        return NULL;
    }
    pProg->memory = (uint32_t*)malloc(pProg->memory_size);

    return pProg;
}

int main(int argc, char **argv){
    if(argc < 2){
        fprintf(stderr, "Incorrect Usage:\nInput file must be specified\n");
        exit(1);
    }

    return 0;
}
