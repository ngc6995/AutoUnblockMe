import auto_unblock_me as aum

print("+-----------------------------+")
print("| Automatic Unblock Me Solver |")
print("+-----------------------------+\n")

print("Step 1 - Take Screenshot ... ", end='')
screenshot, game_window_topleft = aum.take_screenshot()

if screenshot is not None:
    print("Ok!")
    grid_image = aum.crop_grid_image(screenshot)
    print("Step 2 - Extract blocks information ... ", end='')
    blocks = aum.extract_blocks(grid_image)

    if blocks is not None:
        print("Ok!")
        #show_blocks_detected(grid_image, blocks)
        start_grid = aum.map_blocks_to_grid(blocks)
        print("Step 3 - Searching solution ... ", end='')
        path = aum.find_path_astar(start_grid)

        if path:
            moves = aum.get_moves(path)
            print(f"Ok!\n")
            print(f"Total {len(moves)} moves to solve this puzzle.\n")
            aum.play_moves(game_window_topleft, path, moves)
            print(">>> Finish <<<")
        else:
            print(">>> No solution <<<")
    else:
        print("Fail!")
        print(">>> The screenshot is not a puzzle <<<")
else:
    print("Fail!")
    print(">>> The app Unblock Me Free is not running <<<")
