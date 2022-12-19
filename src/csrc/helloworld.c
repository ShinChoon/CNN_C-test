// #include <stdio.h>
// #include <stdlib.h>
// #include <stdint.h>
// unsigned char *convert_to_hex(float number);
// float convert_to_number(const unsigned char *mystring);

// int main(){
//     unsigned char *mystring = convert_to_hex(1.234504);
//     printf("\n");

//     float number = convert_to_number(mystring);
//     printf("Converted back: \n%f\n", number);

//     return 0;
// }

// unsigned char *convert_to_hex(float number)
// {
//     union
//     {
//         float f;
//         unsigned char b[sizeof(float)];
//     } v = {number};
//     size_t i;
//     unsigned char *answer = (unsigned char *)malloc(sizeof(unsigned char *) * sizeof(float));
//     for (i = 0; i < sizeof(v.b); ++i)
//     {
//         answer[i] = v.b[sizeof(v.b)-i-1]; 
//     }

//     return answer;
// }

// float convert_to_number(const unsigned char *mystring)
// {
//         uint32_t num;
//         unsigned char temp[sizeof(float)+1];
//         unsigned char readingreg[sizeof(float)];
//         unsigned char answer[] = "0x";
//         int i = 0;
//         while (i < sizeof(float))
//         {
//             sprintf(temp + i * 2, "%02x", mystring[i]);
//             i++;
//         }
//         strcat(answer, temp);
//         printf("@@@: %s\n", answer);

//         float f;
//         sscanf(temp, "%x", &num); // assuming you checked input
//         f = *((float *)&num);
//         return f;
// }