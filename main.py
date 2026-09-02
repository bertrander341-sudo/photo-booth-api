# 2. Reemplazo de fondo profesional por inundación desde los bordes (FloodFill)
        if bg_color != "none":
            bg_colors_map = {
                "white": (255, 255, 255),
                "black": (0, 0, 0),
                "yellow": (0, 255, 255),
                "red": (0, 0, 255),
                "blue": (255, 0, 0),
                "green": (0, 255, 0),
                "gray": (128, 128, 128)
            }
            target_bgr = bg_colors_map.get(bg_color, (255, 255, 255))
            
            h_img, w_img = frame.shape[:2]
            # Crear máscara para floodfill (requiere 2 píxeles más por lado)
            flood_mask = np.zeros((h_img + 2, w_img + 2), np.uint8)
            
            # Aplicar floodfill desde las 4 esquinas exteriores para asegurar el fondo completo
            cv2.floodFill(frame, flood_mask, (0, 0), target_bgr, (20, 20, 20), (20, 20, 20), cv2.FLOODFILL_FIXED_RANGE)
            cv2.floodFill(frame, flood_mask, (w_img - 1, 0), target_bgr, (20, 20, 20), (20, 20, 20), cv2.FLOODFILL_FIXED_RANGE)
            cv2.floodFill(frame, flood_mask, (0, h_img - 1), target_bgr, (20, 20, 20), (20, 20, 20), cv2.FLOODFILL_FIXED_RANGE)
            cv2.floodFill(frame, flood_mask, (w_img - 1, h_img - 1), target_bgr, (20, 20, 20), (20, 20, 20), cv2.FLOODFILL_FIXED_RANGE)    
