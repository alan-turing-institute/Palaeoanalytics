This is an example of the output data with comments to 
understand the variables:

```json
{
   "id":"rub_al_khali", // name of the image
   "conversion_px":0.040, // conversion from pixel to mm
   "n_surfaces":4, // number of outer surfaces found
   "lithic_contours":[
      {
         "surface_id":0, // largest surface id
         "classification":"Ventral", // surface classification
         "total_area_px":515662.0, // total area of surface in pixels
         "total_area":808.2, // total area of surface in mm
         "max_breadth":22.0, // surface maximum breadth
         "max_length":53.6, // surface maximum lengh
         "polygon_count":7, // numer of vertices measured in an approximate polygon fitted to the surface
         "scar_count":0, // number of scars in that surface
         "percentage_detected_scars":0.0, // percentage of the surface that contains scars
         "scar_contours":[ // empty scar count
         ]
      },
      {
         "surface_id":1, // second largest surface id
         "classification":"Dorsal",
         "total_area_px":515583.0,
         "total_area":808.0,
         "max_breadth":22.0,
         "max_length":53.6,
         "polygon_count":7,
         "scar_count":5,
         "percentage_detected_scars":0.71,
         "scar_contours":[
            {
               "scar_id":0, // largest scar belonging to surface id = 1
               "total_area_px":139998.0, // total area in pixels of scar
               "total_area":219.4, // total area in mm of scar
               "max_breadth":10.6, // scar maximum breadth
               "max_length":42.1, // scar maximum lenght
               "percentage_of_surface":0.27, // percentage of the scar to the total surface
               "scar_angle":1.74, // angle measured of arrow belonging to that scar
               "polygon_count":5 // numer of vertices measured in an approximate polygon fitted to the scar
            },
            {
               "scar_id":1,
               "total_area_px":111052.5,
               "total_area":174.0,
               "max_breadth":7.6,
               "max_length":43.5,
               "percentage_of_surface":0.22,
               "scar_angle":356.78,
               "polygon_count":6
            },
            {
               "scar_id":2,
               "total_area_px":103554.0,
               "total_area":162.3,
               "max_breadth":6.8,
               "max_length":42.4,
               "percentage_of_surface":0.2,
               "scar_angle":5.6,
               "polygon_count":4
            },
            {
               "scar_id":3,
               "total_area_px":6288.0,
               "total_area":9.9,
               "max_breadth":4.4,
               "max_length":5.9,
               "percentage_of_surface":0.01,
               "scar_angle":"NaN",
               "polygon_count":7
            },
            {
               "scar_id":4,
               "total_area_px":5853.0,
               "total_area":9.2,
               "max_breadth":3.9,
               "max_length":3.4,
               "percentage_of_surface":0.01,
               "scar_angle":"NaN",
               "polygon_count":6
            }
         ]
      },
      {
         "surface_id":2,
         "classification":"Lateral",
         "total_area_px":162660.5,
         "total_area":254.9,
         "max_breadth":8.2,
         "max_length":53.8,
         "polygon_count":3,
         "scar_count":2,
         "percentage_detected_scars":0.47,
         "scar_contours":[
            {
               "scar_id":0,
               "total_area_px":57245.5,
               "total_area":89.7,
               "max_breadth":5.4,
               "max_length":51.5,
               "percentage_of_surface":0.35,
               "scar_angle":"NaN",
               "polygon_count":3
            },
            {
               "scar_id":1,
               "total_area_px":18672.5,
               "total_area":29.3,
               "max_breadth":1.9,
               "max_length":24.6,
               "percentage_of_surface":0.11,
               "scar_angle":"NaN",
               "polygon_count":2
            }
         ]
      },
      {
         "surface_id":3,
         "classification":"Platform",
         "total_area_px":50040.0,
         "total_area":78.4,
         "max_breadth":20.0,
         "max_length":6.3,
         "polygon_count":5,
         "scar_count":0,
         "percentage_detected_scars":0.0,
         "scar_contours":[
         ]
      }
   ]
}
```