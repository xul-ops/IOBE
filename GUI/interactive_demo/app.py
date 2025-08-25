import tkinter as tk
from tkinter import messagebox, filedialog, ttk

import os
import pdb
import cv2
import imageio
import numpy as np
from PIL import Image
from collections import Counter
from interactive_demo.interactions_gui import *
from isutils.edgeEvalPy.nms_process import nms_process_one_image

# from isegm.utils.vis import draw_with_blend_and_clicks
from interactive_demo.isegmtools import draw_with_blend_and_clicks

from interactive_demo.canvas import CanvasImage
from interactive_demo.controller import InteractiveController
from interactive_demo.wrappers import BoundedNumericalEntry, FocusHorizontalScale, FocusCheckButton, FocusButton, FocusLabelFrame


class InteractiveDemoApp(ttk.Frame):

    def __init__(self, master, args, model_path):
        super().__init__(master)
        self.master = master
        master.title("Interactive Edge&Boundary Annotation APP")
        master.withdraw()
        master.update_idletasks()
        x = (master.winfo_screenwidth() - master.winfo_reqwidth()) / 2
        y = (master.winfo_screenheight() - master.winfo_reqheight()) / 2
        master.geometry("+%d+%d" % (x, y))
        self.pack(fill="both", expand=True)

        # self.brs_modes = ['NoBRS', 'RGB-BRS', 'DistMap-BRS', 'f-BRS-A', 'f-BRS-B', 'f-BRS-C']
        self.limit_longest_size = args.limit_longest_size

        self.controller = InteractiveController(model_path, args.device, args,
                                                predictor_params={'brs_mode': 'NoBRS'},
                                                update_image_callback=self._update_image,
                                                load_mask_k=self.load_mask_k,)

        self._init_state()
        self._add_menu()
        self._add_canvas()
        self._add_buttons()

        master.bind('<space>', lambda event: self.controller.finish_object())
        # master.bind('a', lambda event: self.controller.partially_finish_object())

        self.state['zoomin_params']['skip_clicks'].trace(mode='w', callback=self._reset_predictor)
        self.state['zoomin_params']['target_size'].trace(mode='w', callback=self._reset_predictor)
        self.state['zoomin_params']['expansion_ratio'].trace(mode='w', callback=self._reset_predictor)
        # self.state['predictor_params']['net_clicks_limit'].trace(mode='w', callback=self._change_brs_mode)
        # self.state['lbfgs_max_iters'].trace(mode='w', callback=self._change_brs_mode)
        # self._change_brs_mode()

        self.is_radius = 12
        self.image_name = None
        self.image_shape = None
        self.coarse_ep = None
        self.nms_ep = None
        self.filename = None
        self.pred_final = None
        self.flag_final = 0
        self.existing_pre_dir = args.existing_pre_dir
        self.output_dir = args.output_dir 
        self.pos_save_path = args.output_dir + "pos/"
        self.neg_save_path = args.output_dir + "neg/"
        self.pre_save_path = args.output_dir + "pre/"
        self.log_save_path = args.output_dir + "points_log/"
        self.interactive_res_path = args.output_dir + "f_res/"
        if not os.path.exists(self.pos_save_path):
            os.makedirs(self.pos_save_path)
            os.makedirs(self.neg_save_path)
            os.makedirs(self.pre_save_path)
            os.makedirs(self.log_save_path)
            os.makedirs(self.interactive_res_path)
        

    def _init_state(self):
        self.state = {
            'zoomin_params': {
                'use_zoom_in': tk.BooleanVar(value=True),
                'fixed_crop': tk.BooleanVar(value=True),
                'skip_clicks': tk.IntVar(value=-1),
                'target_size': tk.IntVar(value=min(400, self.limit_longest_size)),
                'expansion_ratio': tk.DoubleVar(value=1.4)
            },

            'predictor_params': {
                'net_clicks_limit': tk.IntVar(value=8)
            },
            'brs_mode': tk.StringVar(value='NoBRS'),
            'prob_thresh': tk.DoubleVar(value=0.7),
            'lbfgs_max_iters': tk.IntVar(value=20),

            'alpha_blend': tk.DoubleVar(value=0.5),
            'click_radius': tk.IntVar(value=0),
        }

    def _add_menu(self):
        self.menubar = FocusLabelFrame(self, bd=1)
        self.menubar.pack(side=tk.TOP, fill='x')

        button = FocusButton(self.menubar, text='Load image', command=self._load_image_callback)
        button.pack(side=tk.LEFT)


        # self.save_mask_btn = FocusButton(self.menubar, text='Save mask', command=self._save_mask_callback)
        # self.save_mask_btn.pack(side=tk.LEFT)
        # self.save_mask_btn.configure(state=tk.DISABLED)
        # self.load_mask_btn = FocusButton(self.menubar, text='Load mask', command=self._load_mask_callback)
        # self.load_mask_btn.pack(side=tk.LEFT)
        # self.load_mask_btn.configure(state=tk.DISABLED)
        

        button = FocusButton(self.menubar, text='About', command=self._about_callback)
        button.pack(side=tk.LEFT)
        button = FocusButton(self.menubar, text='Exit', command=self.master.quit)
        button.pack(side=tk.LEFT)

    def _add_canvas(self):
        self.canvas_frame = FocusLabelFrame(self, text="Image")
        self.canvas_frame.rowconfigure(0, weight=1)
        self.canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.canvas_frame, highlightthickness=0, cursor="hand1", width=400, height=400)
        self.canvas.grid(row=0, column=0, sticky='nswe', padx=5, pady=5)

        self.image_on_canvas = None
        self.canvas_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=5, pady=5)

    def _add_buttons(self):
        self.control_frame = FocusLabelFrame(self, text="Controls")
        self.control_frame.pack(side=tk.TOP, fill='x', padx=5, pady=5)
        master = self.control_frame

        # 1 Clicks management
        self.clicks_options_frame = FocusLabelFrame(master, text="Make predictions")
        self.clicks_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        # Make predictions
        self.finish_object_button = FocusButton(self.clicks_options_frame, text='Show me the edges!',
                                                bg='#94d7f2', fg='black', width=25, height=2,
                                                state=tk.NORMAL, command=self.controller.finish_object)
        self.finish_object_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=1)
        # Load previous edge map
        self.load_existing_button = FocusButton(self.clicks_options_frame, text='Load existing edge mask!',
                                                bg='#94d7f2', fg='black', width=25, height=2,
                                                state=tk.NORMAL, command=self._loadExistingOne)
        self.load_existing_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=1)

        self.undo_click_button = FocusButton(self.clicks_options_frame, text='Undo click', bg='#ffe599', fg='black', width=10, height=2,
                        state=tk.DISABLED, command=self.controller.undo_click)  # state=tk.DISABLED
        self.undo_click_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)

        # self.reset_clicks_button = FocusButton(self.clicks_options_frame, text='Reset clicks', bg='#ea9999', fg='black', width=10, height=2,
        #                 state=tk.DISABLED, command=self._reset_last_object)
        # self.reset_clicks_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)


        # # 2 ZoomIn options
        self.zoomin_options_frame = FocusLabelFrame(master, text="ZoomIn options")
        self.zoomin_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusCheckButton(self.zoomin_options_frame, text='Use ZoomIn', command=self._reset_predictor,
                         variable=self.state['zoomin_params']['use_zoom_in']).grid(row=0, column=0, padx=10)
        FocusCheckButton(self.zoomin_options_frame, text='Fixed crop', command=self._reset_predictor,
                         variable=self.state['zoomin_params']['fixed_crop']).grid(row=1, column=0, padx=10)
        tk.Label(self.zoomin_options_frame, text="Skip clicks").grid(row=0, column=1, pady=1, sticky='e')
        tk.Label(self.zoomin_options_frame, text="Target size").grid(row=1, column=1, pady=1, sticky='e')
        tk.Label(self.zoomin_options_frame, text="Expand ratio").grid(row=2, column=1, pady=1, sticky='e')
        BoundedNumericalEntry(self.zoomin_options_frame, variable=self.state['zoomin_params']['skip_clicks'],
                              min_value=-1, max_value=None, vartype=int,
                              name='zoom_in_skip_clicks').grid(row=0, column=2, padx=10, pady=1, sticky='w')
        BoundedNumericalEntry(self.zoomin_options_frame, variable=self.state['zoomin_params']['target_size'],
                              min_value=100, max_value=self.limit_longest_size, vartype=int,
                              name='zoom_in_target_size').grid(row=1, column=2, padx=10, pady=1, sticky='w')
        BoundedNumericalEntry(self.zoomin_options_frame, variable=self.state['zoomin_params']['expansion_ratio'],
                              min_value=1.0, max_value=2.0, vartype=float,
                              name='zoom_in_expansion_ratio').grid(row=2, column=2, padx=10, pady=1, sticky='w')
        self.zoomin_options_frame.columnconfigure((0, 1, 2), weight=1)


        # 2 Point & encoding
        self.point_encoding = FocusLabelFrame(master, text="One Point & Encoding (Disk/Distance)")
        self.point_encoding.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        # encoding_1
        self.encoding_1 = FocusButton(self.point_encoding, text='Disk pos', bg='#b6d7a8', fg='black', width=7, height=2,
                                      state=tk.NORMAL, command=self._point_disk)
        self.encoding_1.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=1)
        self.encoding_1_n = FocusButton(self.point_encoding, text='Disk neg', bg='#dfd2f7', fg='black', width=7, height=2,
                                      state=tk.NORMAL, command=self._point_disk_n)
        self.encoding_1_n.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=1)

        # encoding_2
        self.encoding_2 = FocusButton(self.point_encoding, text='EucDis pos', bg='#ffe599', fg='black', width=7, height=2,
                                      state=tk.NORMAL, command=self._point_eudis)
        self.encoding_2.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=1)
        self.encoding_2_n = FocusButton(self.point_encoding, text='EucDis neg', bg='#dfd2f7', fg='black', width=7, height=2,
                                      state=tk.NORMAL, command=self._point_eudis_n)
        self.encoding_2_n.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=1)
        # encoding_3
        self.encoding_3 = FocusButton(self.point_encoding, text='GauDis pos', bg='#ea9999', fg='black', width=7, height=2,
                                      state=tk.NORMAL, command=self._point_gaudis)
        self.encoding_3.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=1)
        self.encoding_3_n = FocusButton(self.point_encoding, text='GauDis neg', bg='#dfd2f7', fg='black', width=7, height=2,
                                      state=tk.NORMAL, command=self._point_gaudis_n)
        self.encoding_3_n.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=1)

        # 3 two Point & encoding
        self.point2_encoding = FocusLabelFrame(master, text="Two Points & Encoding (Disk/Distance/Synthetic Scribbles)")
        self.point2_encoding.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        # encoding_1
        self.encoding_1 = FocusButton(self.point2_encoding, text='Syn Scribble pos', bg='#b6d7a8', fg='black', width=15, height=2,
                                      state=tk.NORMAL, command=self._point2_synscr)
        self.encoding_1.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=1)
        self.encoding_1_n = FocusButton(self.point2_encoding, text='Syn Scribble neg', bg='#dfd2f7', fg='black', width=15, height=2,
                                        state=tk.NORMAL, command=self._point2_synscr_n)
        self.encoding_1_n.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=1)

        # # encoding_2
        # self.encoding_2 = FocusButton(self.point2_encoding, text='EucD p', bg='#ffe599', fg='black', width=5, height=2,
        #                               state=tk.NORMAL, command=self._point2_eudis)
        # self.encoding_2.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=1)
        # self.encoding_2_n = FocusButton(self.point2_encoding, text='EucD n', bg='#dfd2f7', fg='black', width=4, height=2,
        #                                 state=tk.NORMAL, command=self._point2_eudis_n)
        # self.encoding_2_n.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=1)
        # # encoding_3
        # self.encoding_3 = FocusButton(self.point2_encoding, text='GauD p', bg='#ea9999', fg='black', width=5, height=2,
        #                               state=tk.NORMAL, command=self._point2_gaudis)
        # self.encoding_3.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=1)
        # self.encoding_3_n = FocusButton(self.point2_encoding, text='GauD n', bg='#dfd2f7', fg='black', width=4, height=2,
        #                                 state=tk.NORMAL, command=self._point2_gaudis_n)
        # self.encoding_3_n.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=1)

        # 4 scribble & encoding
        self.scribble_encoding = FocusLabelFrame(master, text="Scribble & Encoding (Disk/Distance)")
        self.scribble_encoding.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        # encoding_1
        self.encoding_1 = FocusButton(self.scribble_encoding, text='Scribble pos', bg='#b6d7a8', fg='black', width=15, height=2,
                                      state=tk.NORMAL, command=self._scribble_disk)
        self.encoding_1.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=1)
        self.encoding_1_n = FocusButton(self.scribble_encoding, text='Scribble neg', bg='#dfd2f7', fg='black', width=15, height=2,
                                        state=tk.NORMAL, command=self._scribble_disk_n)
        self.encoding_1_n.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=1)

        # # encoding_2
        # self.encoding_2 = FocusButton(self.scribble_encoding, text='EucD p', bg='#ffe599', fg='black', width=5, height=2,
        #                               state=tk.NORMAL, command=self._scribble_eudis)
        # self.encoding_2.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=1)
        # self.encoding_2_n = FocusButton(self.scribble_encoding, text='EucD n', bg='#dfd2f7', fg='black', width=4, height=2,
        #                                 state=tk.NORMAL, command=self._scribble_eudis_n)
        # self.encoding_2_n.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=1)
        # # encoding_3
        # self.encoding_3 = FocusButton(self.scribble_encoding, text='GauD p', bg='#ea9999', fg='black', width=5, height=2,
        #                               state=tk.NORMAL, command=self._scribble_gaudis)
        # self.encoding_3.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=1)
        # self.encoding_3_n = FocusButton(self.scribble_encoding, text='GauD n', bg='#dfd2f7', fg='black', width=4, height=2,
        #                                 state=tk.NORMAL, command=self._scribble_gaudis_n)
        # self.encoding_3_n.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=1)



        # 5 visualization
        self.visualization = FocusLabelFrame(master, text="Visualization")
        self.visualization.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)

        # nms button
        self.nmsButton = FocusButton(self.visualization, text='NMS', bg='#b6d7a8', fg='black',
                                     width=8, height=2,
                                     state=tk.NORMAL, command=self._nms)
        self.nmsButton.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=1)

        # threshold button
        self.thresholdButton = FocusButton(self.visualization, text='Visualize NMS Edge',
                                           bg='#b6d7a8', fg='black',
                                           width=16, height=2,
                                           state=tk.NORMAL, command=self._showVis)
        self.thresholdButton.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=1)


        # 6 manual modification
        self.manual = FocusLabelFrame(master, text="Manual modification")
        self.manual.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        # naive add button
        self.naiveAddButton = FocusButton(self.manual, text='naive add', bg='#b6d7a8', fg='black',
                                     width=8, height=2,
                                     state=tk.NORMAL, command=self._naiveAdd)
        self.naiveAddButton.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=1)
        # naive eraser button
        self.naiveEraserButton = FocusButton(self.manual, text='naive \n eraser',
                                           bg='#b6d7a8', fg='black',
                                           width=8, height=2,
                                           state=tk.NORMAL, command=self._naiveEraser)
        self.naiveEraserButton.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=1)

        # save final edge map button
        self.saveFinalButton = FocusButton(self.manual, text='save \n edge map',
                                             bg='#b6d7a8', fg='black',
                                             width=8, height=2,
                                             state=tk.NORMAL, command=self._saveFinalEdgemap)
        self.saveFinalButton.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=1)



        # # 3 BRS options
        # self.brs_options_frame = FocusLabelFrame(master, text="BRS options")
        # self.brs_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        # menu = tk.OptionMenu(self.brs_options_frame, self.state['brs_mode'],
        #                      *self.brs_modes, command=self._change_brs_mode)
        # menu.config(width=11)
        # menu.grid(rowspan=2, column=0, padx=10)
        # self.net_clicks_label = tk.Label(self.brs_options_frame, text="Network clicks")
        # self.net_clicks_label.grid(row=0, column=1, pady=2, sticky='e')
        # self.net_clicks_entry = BoundedNumericalEntry(self.brs_options_frame,
        #                                               variable=self.state['predictor_params']['net_clicks_limit'],
        #                                               min_value=0, max_value=None, vartype=int, allow_inf=True,
        #                                               name='net_clicks_limit')
        # self.net_clicks_entry.grid(row=0, column=2, padx=10, pady=2, sticky='w')
        # self.lbfgs_iters_label = tk.Label(self.brs_options_frame, text="L-BFGS\nmax iterations")
        # self.lbfgs_iters_label.grid(row=1, column=1, pady=2, sticky='e')
        # self.lbfgs_iters_entry = BoundedNumericalEntry(self.brs_options_frame, variable=self.state['lbfgs_max_iters'],
        #                                                min_value=1, max_value=1000, vartype=int,
        #                                                name='lbfgs_max_iters')
        # self.lbfgs_iters_entry.grid(row=1, column=2, padx=10, pady=2, sticky='w')
        # self.brs_options_frame.columnconfigure((0, 1), weight=1)

        # 4 params
        # edge probility threhold 
        self.prob_thresh_frame = FocusLabelFrame(master, text="Choose edge probility threshold")
        self.prob_thresh_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.prob_thresh_frame, from_=0.0, to=1.0, command=self._update_prob_thresh,
                             variable=self.state['prob_thresh']).pack(padx=10)
        # res alpha in original image
        self.alpha_blend_frame = FocusLabelFrame(master, text="Alpha blending coefficient")
        self.alpha_blend_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.alpha_blend_frame, from_=0.0, to=1.0, command=self._update_blend_alpha,
                             variable=self.state['alpha_blend']).pack(padx=10, anchor=tk.CENTER)
        # naive eraser radius 
        self.click_radius_frame = FocusLabelFrame(master, text="Navie eraser radius")
        self.click_radius_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.click_radius_frame, from_=0, to=7, resolution=1, command=self._update_click_radius,
                             variable=self.state['click_radius']).pack(padx=10, anchor=tk.CENTER)


    def _load_image_callback(self):
        self.menubar.focus_set()
        if self._check_entry(self):
            filename = filedialog.askopenfilename(parent=self.master, filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*"),
            ], title="Chose an image")
            self.filename = filename

            if len(filename) > 0:
                image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
                self.image_name = filename
                self.image_shape = image.shape
                print('image_name:{} \nimage_shape:{}'.format(self.image_name, self.image_shape))
                self.controller.set_image(image, self.image_name)
                self._update_image()
                # self.save_mask_btn.configure(state=tk.NORMAL)
                # self.load_mask_btn.configure(state=tk.NORMAL)

    # def _save_mask_callback(self):
    #     self.menubar.focus_set()
    #     if self._check_entry(self):
    #         mask = self.controller.result_mask
    #         if mask is None:
    #             return

    #         filename = filedialog.asksaveasfilename(parent=self.master, initialfile='mask.png', filetypes=[
    #             ("PNG image", "*.png"),
    #             ("BMP image", "*.bmp"),
    #             ("All files", "*.*"),
    #         ], title="Save the current mask as...")

    #         if len(filename) > 0:
    #             if mask.max() < 256:
    #                 mask = mask.astype(np.uint8)
    #                 mask *= 255 // mask.max()
    #             cv2.imwrite(filename, mask)

    # def _load_mask_callback(self):
    #     if not self.controller.net.with_prev_mask:
    #         messagebox.showwarning("Warning", "The current model doesn't support loading external masks. "
    #                                           "Please use ITER-M models for that purpose.")
    #         return

    #     self.menubar.focus_set()
    #     if self._check_entry(self):
    #         filename = filedialog.askopenfilename(parent=self.master, filetypes=[
    #             ("Binary mask (png, bmp)", "*.png *.bmp"),
    #             ("All files", "*.*"),
    #         ], title="Chose an image")

    #         if len(filename) > 0:
    #             mask = cv2.imread(filename)[:, :, 0] > 127
    #             self.controller.set_mask(mask)
    #             self._update_image()

    def _about_callback(self):
        self.menubar.focus_set()

        text = [
            "Interactive Annotation Application",
            "Undo click functions has not been re-added",
            "2023"
        ]

        messagebox.showinfo("About Demo", '\n'.join(text))

    def _reset_last_object(self):
        self.state['alpha_blend'].set(0.5)
        self.state['prob_thresh'].set(0.7)
        self.controller.reset_last_object()

    def _update_prob_thresh(self, value):
        if self.controller.is_incomplete_mask:
            self.controller.prob_thresh = self.state['prob_thresh'].get()
            self._update_image()

    def _update_blend_alpha(self, value):
        self._update_image()

    def _update_click_radius(self, *args):
        if self.image_on_canvas is None:
            return
        # self._update_image()

    # def _change_brs_mode(self, *args):
    #     if self.state['brs_mode'].get() == 'NoBRS':
    #         self.net_clicks_entry.set('INF')
    #         self.net_clicks_entry.configure(state=tk.DISABLED)
    #         self.net_clicks_label.configure(state=tk.DISABLED)
    #         self.lbfgs_iters_entry.configure(state=tk.DISABLED)
    #         self.lbfgs_iters_label.configure(state=tk.DISABLED)
    #     else:
    #         if self.net_clicks_entry.get() == 'INF':
    #             self.net_clicks_entry.set(8)
    #         self.net_clicks_entry.configure(state=tk.NORMAL)
    #         self.net_clicks_label.configure(state=tk.NORMAL)
    #         self.lbfgs_iters_entry.configure(state=tk.NORMAL)
    #         self.lbfgs_iters_label.configure(state=tk.NORMAL)

    #     self._reset_predictor()

    def _reset_predictor(self, *args, **kwargs):
        brs_mode = self.state['brs_mode'].get()
        prob_thresh = self.state['prob_thresh'].get()
        net_clicks_limit = None if brs_mode == 'NoBRS' else self.state['predictor_params']['net_clicks_limit'].get()

        if self.state['zoomin_params']['use_zoom_in'].get():
            zoomin_params = {
                'skip_clicks': self.state['zoomin_params']['skip_clicks'].get(),
                'target_size': self.state['zoomin_params']['target_size'].get(),
                'expansion_ratio': self.state['zoomin_params']['expansion_ratio'].get()
            }
            if self.state['zoomin_params']['fixed_crop'].get():
                zoomin_params['target_size'] = (zoomin_params['target_size'], zoomin_params['target_size'])
        else:
            zoomin_params = None

        predictor_params = {
            'brs_mode': brs_mode,
            'prob_thresh': prob_thresh,
            'zoom_in_params': zoomin_params,
            'predictor_params': {
                'net_clicks_limit': net_clicks_limit,
                'max_size': self.limit_longest_size
            },
            'brs_opt_func_params': {'min_iou_diff': 1e-3},
            'lbfgs_params': {'maxfun': self.state['lbfgs_max_iters'].get()}
        }
        self.controller.reset_predictor(predictor_params)

    def _click_callback(self, is_positive, x, y):
        self.canvas.focus_set()
        if self.image_on_canvas is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        if self._check_entry(self):
            if is_positive == 1:
                print("Left scribble/Edge ({}, {})".format(x, y))
            else:
                print("Right scribble/Non-edge ({}, {})".format(x, y))
            self.controller.add_click(x, y, is_positive)
            # self._update_image()


    def _update_image(self, reset_canvas=False):
        image = self.controller.get_visualization(alpha_blend=self.state['alpha_blend'].get(),
                                                  click_radius=self.state['click_radius'].get())
        
        if self.image_on_canvas is None:
            self.image_on_canvas = CanvasImage(self.canvas_frame, self.canvas, self.image_name, self.image_shape[:2], self.output_dir)
            self.image_on_canvas.register_click_callback(self._click_callback)

        # self._set_click_dependent_widgets_state()
        if image is not None:
            self.image_on_canvas.reload_image(Image.fromarray(image), reset_canvas)
            
    # def _update_image(self, reset_canvas=False):
    #     image = self.controller.probs_history
    #     if image is not None:
    #         self.image_on_canvas.reload_image(Image.fromarray(image), reset_canvas)

    # def _set_click_dependent_widgets_state(self):
    #     after_1st_click_state = tk.NORMAL if self.controller.is_incomplete_mask else tk.DISABLED
    #     before_1st_click_state = tk.DISABLED if self.controller.is_incomplete_mask else tk.NORMAL
    #
    #     self.finish_object_button.configure(state=after_1st_click_state)
    #     self.undo_click_button.configure(state=after_1st_click_state)
    #     self.reset_clicks_button.configure(state=after_1st_click_state)
    #     self.zoomin_options_frame.set_frame_state(before_1st_click_state)
    #     self.brs_options_frame.set_frame_state(before_1st_click_state)
    #
    #     if self.state['brs_mode'].get() == 'NoBRS':
    #         self.net_clicks_entry.configure(state=tk.DISABLED)
    #         self.net_clicks_label.configure(state=tk.DISABLED)
    #         self.lbfgs_iters_entry.configure(state=tk.DISABLED)
    #         self.lbfgs_iters_label.configure(state=tk.DISABLED)

    def load_mask_k(self, pred):
        # self.image_on_canvas.reload_image(Image.fromarray(pred), reset_canvas=True)
        self.controller.set_image(pred, self.image_name, vis=True)

    def _check_entry(self, widget):
        all_checked = True
        if widget.winfo_children is not None:
            for w in widget.winfo_children():
                all_checked = all_checked and self._check_entry(w)

        if getattr(widget, "_check_bounds", None) is not None:
            all_checked = all_checked and widget._check_bounds(widget.get(), '-1')

        return all_checked


    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&   interaction & encoding   &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # @staticmethod
    def save_interaction_image(self, pos, save_path, image_shape, is_positive=True):
        # print(save_path)
        # print(self.image_name)
        previous_pos = cv2.imread(save_path, cv2.IMREAD_GRAYSCALE)
        if previous_pos is None:
            previous_pos = np.zeros(image_shape[:2])
        # print(self.image_shape)
        previous_pos = np.array(previous_pos)
        pos = np.array(pos)
        print("pos.shape:{}, previous_pos.shape:{}".format(pos.shape, previous_pos.shape))       
        pos = pos + previous_pos
        pos[np.where(pos > 255)] = 255
        cv2.imwrite(save_path, pos)
        if self.controller._init_mask is not None:
            pep_pred = self.controller._init_mask
            if is_positive:
                pos = pos.reshape((image_shape[0], image_shape[1], 1)) / 255
                neg = pep_pred[:, : , 1].reshape((image_shape[0], image_shape[1], 1))
            else:
                neg = pos.reshape((image_shape[0], image_shape[1], 1)) / 255
                pos = pep_pred[:, : , 0].reshape((image_shape[0], image_shape[1], 1))
        else:
            if is_positive:
                pos = pos.reshape((image_shape[0], image_shape[1], 1)) /255
                neg = np.zeros((image_shape[0], image_shape[1], 1))
            else:
                neg = pos.reshape((image_shape[0], image_shape[1], 1)) /255  
                pos = np.zeros((image_shape[0], image_shape[1], 1))   

        if self.nms_ep is not None:
            pre = self.nms_ep.reshape((image_shape[0], image_shape[1], 1))
        else:
            if self.controller.final_pred is not None:
                self._nms()
                pre = self.nms_ep.reshape((image_shape[0], image_shape[1], 1))
            else:
                pre = np.zeros((image_shape[0], image_shape[1], 1))
        # print(pos.shape, neg.shape, pre.shape)
        pep_pred = np.concatenate((pos, neg, pre), axis=-1).astype(np.float32)
        self.controller.set_mask(pep_pred)
            

    def _point_disk(self):
        print('Interaction = point, Encoding = disk')
        log_name = self.log_save_path + self.image_name.split('/')[-1][:-4] + ".txt"
        pos = generate_interaction_maps(self.image_on_canvas.user_click, self.image_on_canvas.image_shape,
                                        interaction='point', encoding='disk',
                                        radius=self.is_radius, gausigma=0.04, save_log=log_name, op_way='middle', num_points=1)
                               
        save_path = self.pos_save_path + self.image_name.split('/')[-1][:-4] + '_pos.png'
        
        # previous_pos = cv2.imread(save_path, cv2.IMREAD_GRAYSCALE)
        # if previous_pos is None:
        #     previous_pos = np.zeros(self.image_shape[:2])
        # previous_pos = np.array(previous_pos)
        # pos = np.array(pos)
        # print("pos.shape:{}, previous_pos.shape:{}".format(pos.shape, previous_pos.shape))
        
        # pos = pos + previous_pos
        # pos[np.where(pos > 255)] = 255
        # cv2.imwrite(save_path, pos)

        self.save_interaction_image(pos, save_path, self.image_shape)
        self.image_on_canvas.user_click = []
        
    def _point_disk_n(self):
        print('Interaction = point, Encoding = disk')
        log_name = self.log_save_path + self.image_name.split('/')[-1][:-4] + ".txt"
        pos = generate_interaction_maps(self.image_on_canvas.user_click, self.image_on_canvas.image_shape,
                                        interaction='point', encoding='disk',
                                        radius=self.is_radius, gausigma=0.04, save_log=log_name, op_way='middle', num_points=1)
        save_path = self.neg_save_path + self.image_name.split('/')[-1][:-4] + '_neg.png'
        self.save_interaction_image(pos, save_path, self.image_shape, is_positive=False)
        self.image_on_canvas.user_click = []
        
    def _point_eudis(self):
        print('Interaction = point, Encoding = Euclidean distance')
        log_name = self.log_save_path + self.image_name.split('/')[-1][:-4] + ".txt"
        pos = generate_interaction_maps(self.image_on_canvas.user_click, self.image_on_canvas.image_shape,
                                        interaction='point', encoding='eudistance',
                                        radius=self.is_radius, gausigma=0.04, save_log=log_name, op_way='middle', num_points=1)
        save_path = self.pos_save_path + self.image_name.split('/')[-1][:-4] + '_pos.png'
        self.save_interaction_image(pos, save_path, self.image_shape)
        self.image_on_canvas.user_click = []
        
    def _point_eudis_n(self):
        print('Interaction = point, Encoding = Euclidean distance')
        log_name = self.log_save_path + self.image_name.split('/')[-1][:-4] + ".txt"        
        pos = generate_interaction_maps(self.image_on_canvas.user_click, self.image_on_canvas.image_shape,
                                        interaction='point', encoding='eudistance',
                                        radius=self.is_radius, gausigma=0.04, save_log=log_name, op_way='middle', num_points=1)
        save_path = self.neg_save_path + self.image_name.split('/')[-1][:-4] + '_neg.png'
        self.save_interaction_image(pos, save_path, self.image_shape, is_positive=False)
        self.image_on_canvas.user_click = []
        
    def _point_gaudis(self):
        print('Interaction = point, Encoding = Gaussian distance')
        log_name = self.log_save_path + self.image_name.split('/')[-1][:-4] + ".txt"
        pos = generate_interaction_maps(self.image_on_canvas.user_click, self.image_on_canvas.image_shape,
                                        interaction='point', encoding='gaudistance',
                                        radius=self.is_radius, gausigma=0.04, save_log=log_name, op_way='middle', num_points=1)
        save_path = self.pos_save_path + self.image_name.split('/')[-1][:-4] + '_pos.png'
        self.save_interaction_image(pos, save_path, self.image_shape)
        self.image_on_canvas.user_click = []
        
    def _point_gaudis_n(self):
        print('Interaction = point, Encoding = Gaussian distance')
        log_name = self.log_save_path + self.image_name.split('/')[-1][:-4] + ".txt"
        pos = generate_interaction_maps(self.image_on_canvas.user_click, self.image_on_canvas.image_shape,
                                        interaction='point', encoding='gaudistance',
                                        radius=self.is_radius, gausigma=0.04, save_log=log_name, op_way='middle', num_points=1)
        save_path = self.neg_save_path  + self.image_name.split('/')[-1][:-4] + '_neg.png'
        self.save_interaction_image(pos, save_path, self.image_shape, is_positive=False)
        self.image_on_canvas.user_click = []


    def _scribble_disk(self):
        print('Interaction = scribbles, Encoding = scribbles')
        log_name = self.log_save_path + self.image_name.split('/')[-1][:-4] + ".txt"
        pos = generate_interaction_maps([self.image_on_canvas.user_click], self.image_on_canvas.image_shape,
                                        interaction='scribbles', encoding='scribbles',
                                        radius=self.is_radius, gausigma=0.04, save_log=log_name, op_way='middle', num_points=1)
        save_path = self.pos_save_path + self.image_name.split('/')[-1][:-4] + '_pos.png'
        self.save_interaction_image(pos, save_path, self.image_shape)
        self.image_on_canvas.user_click = []
        
    def _scribble_disk_n(self):
        print('Interaction = scribbles, Encoding = scribbles')
        # print(self.image_on_canvas.user_click)
        # print(self.image_on_canvas.image_shape)
        log_name = self.log_save_path + self.image_name.split('/')[-1][:-4] + ".txt"
        pos = generate_interaction_maps([self.image_on_canvas.user_click], self.image_on_canvas.image_shape,
                                        interaction='scribbles', encoding='scribbles',
                                        radius=self.is_radius, gausigma=0.04, save_log=log_name, op_way='middle', num_points=1)
        save_path = self.neg_save_path + self.image_name.split('/')[-1][:-4] + '_neg.png'
        self.save_interaction_image(pos, save_path, self.image_shape, is_positive=False)
        self.image_on_canvas.user_click = []
        
    def _scribble_eudis(self):
        print('Interaction = scribbles, Encoding = Euclidean distance')
        log_name = self.log_save_path + self.image_name.split('/')[-1][:-4] + ".txt"
        pos = generate_interaction_maps([self.image_on_canvas.user_click], self.image_on_canvas.image_shape,
                                        interaction='scribbles', encoding='eudistance',
                                        radius=self.is_radius, gausigma=0.04, save_log=log_name, op_way='middle', num_points=1)

        save_path = self.pos_save_path + self.image_name.split('/')[-1][:-4] + '_pos.png'
        self.save_interaction_image(pos, save_path, self.image_shape)
        self.image_on_canvas.user_click = []
        
    def _scribble_eudis_n(self):
        print('Interaction = scribbles, Encoding = Euclidean distance')
        log_name = self.log_save_path + self.image_name.split('/')[-1][:-4] + ".txt"
        pos = generate_interaction_maps([self.image_on_canvas.user_click], self.image_on_canvas.image_shape,
                                        interaction='scribbles', encoding='eudistance',
                                        radius=self.is_radius, gausigma=0.04, save_log=log_name, op_way='middle', num_points=1)
        save_path = self.neg_save_path + self.image_name.split('/')[-1][:-4] + '_neg.png'
        self.save_interaction_image(pos, save_path, self.image_shape, is_positive=False)
        self.image_on_canvas.user_click = []
        
    def _scribble_gaudis(self):
        print('Interaction = scribbles, Encoding = Gaussian distance')
        log_name = self.log_save_path + self.image_name.split('/')[-1][:-4] + ".txt"
        pos = generate_interaction_maps([self.image_on_canvas.user_click], self.image_on_canvas.image_shape,
                                        interaction='scribbles', encoding='gaudistance',
                                        radius=self.is_radius, gausigma=0.04, save_log=log_name, op_way='middle', num_points=1)
        save_path = self.pos_save_path + self.image_name.split('/')[-1][:-4] + '_pos.png'
        self.save_interaction_image(pos, save_path, self.image_shape)
        self.image_on_canvas.user_click = []
        
    def _scribble_gaudis_n(self):
        print('Interaction = scribbles, Encoding = Gaussian distance')
        log_name = self.log_save_path + self.image_name.split('/')[-1][:-4] + ".txt"
        pos = generate_interaction_maps([self.image_on_canvas.user_click], self.image_on_canvas.image_shape,
                                        interaction='scribbles', encoding='gaudistance',
                                        radius=self.is_radius, gausigma=0.04, save_log=log_name, op_way='middle', num_points=1)
        save_path = self.neg_save_path + self.image_name.split('/')[-1][:-4] + '_neg.png'
        self.save_interaction_image(pos, save_path, self.image_shape, is_positive=False)
        self.image_on_canvas.user_click = []


    def _point2_synscr(self):
        print('Interaction = point 2, Encoding = synscr')
        log_name = self.log_save_path + self.image_name.split('/')[-1][:-4] + ".txt"
        pos = generate_interaction_maps([self.image_on_canvas.user_click], self.image_on_canvas.image_shape,
                                        interaction='syn_scribbles', encoding='syn_scribbles',
                                        radius=self.is_radius, gausigma=0.04, save_log=log_name, op_way='middle', num_points=1)
        save_path = self.pos_save_path + self.image_name.split('/')[-1][:-4] + '_pos.png'
        self.save_interaction_image(pos, save_path, self.image_shape)
        self.image_on_canvas.user_click.clear()  # = []

        
    def _point2_synscr_n(self):
        print('Interaction = point 2, Encoding = synscr')
        log_name = self.log_save_path + self.image_name.split('/')[-1][:-4] + ".txt"
        pos = generate_interaction_maps([self.image_on_canvas.user_click], self.image_on_canvas.image_shape,
                                        interaction='syn_scribbles', encoding='syn_scribbles',
                                        radius=self.is_radius, gausigma=0.04, save_log=log_name, op_way='middle', num_points=1)
        save_path = self.neg_save_path + self.image_name.split('/')[-1][:-4] + '_neg.png'
        self.save_interaction_image(pos, save_path, self.image_shape, is_positive=False)
        self.image_on_canvas.user_click = []

    def _point2_eudis(self):
        print('Interaction = syn_scribbles, Encoding = Euclidean distance')
        log_name = self.log_save_path + self.image_name.split('/')[-1][:-4] + ".txt"
        pos = generate_interaction_maps([self.image_on_canvas.user_click], self.image_on_canvas.image_shape,
                                        interaction='syn_scribbles', encoding='syn_scribbles_eudis',
                                        radius=self.is_radius, gausigma=0.04, save_log=log_name, op_way='middle', num_points=1)
        save_path = self.pos_save_path + self.image_name.split('/')[-1][:-4] + '_pos.png'
        self.save_interaction_image(pos, save_path, self.image_shape)
        self.image_on_canvas.user_click = []
        
    def _point2_eudis_n(self):
        print('Interaction = syn_scribbles, Encoding = Euclidean distance')
        log_name = self.log_save_path + self.image_name.split('/')[-1][:-4] + ".txt"
        pos = generate_interaction_maps([self.image_on_canvas.user_click], self.image_on_canvas.image_shape,
                                        interaction='syn_scribbles', encoding='syn_scribbles_eudis',
                                        radius=self.is_radius, gausigma=0.04, save_log=log_name, op_way='middle', num_points=1)
        save_path = self.neg_save_path +self.image_name.split('/')[-1][:-4] + '_neg.png'
        self.save_interaction_image(pos, save_path, self.image_shape, is_positive=False)
        self.image_on_canvas.user_click = []
        
    def _point2_gaudis(self):
        print('Interaction = syn_scribbles, Encoding = Gaussian distance')
        log_name = self.log_save_path + self.image_name.split('/')[-1][:-4] + ".txt"
        pos = generate_interaction_maps([self.image_on_canvas.user_click], self.image_on_canvas.image_shape,
                                        interaction='syn_scribbles', encoding='syn_scribbles_gaudis',
                                        radius=self.is_radius, gausigma=0.04, save_log=log_name, op_way='middle', num_points=1)
        save_path = self.pos_save_path + self.image_name.split('/')[-1][:-4] + '_pos.png'
        self.save_interaction_image(pos, save_path, self.image_shape)
        self.image_on_canvas.user_click = []
        
    def _point2_gaudis_n(self):
        print('Interaction = syn_scribbles, Encoding = Gaussian distance')
        log_name = self.log_save_path + self.image_name.split('/')[-1][:-4] + ".txt"
        pos = generate_interaction_maps([self.image_on_canvas.user_click], self.image_on_canvas.image_shape,
                                        interaction='syn_scribbles', encoding='syn_scribbles_gaudis',
                                        radius=self.is_radius, gausigma=0.04, save_log=log_name, op_way='middle', num_points=1)
        save_path = self.neg_save_path + self.image_name.split('/')[-1][:-4] + '_neg.png'
        self.save_interaction_image(pos, save_path, self.image_shape, is_positive=False)
        self.image_on_canvas.user_click = []

    def _naiveAdd(self):
        # threshold = self.state['prob_thresh'].get()
        # pred = (self.controller.final_pred > threshold).astype(int)
        pred = self.pred_final
        naiveAdd = np.zeros(self.image_shape[:2])
        # print(self.image_on_canvas.user_click)
        # naiveAdd[self.image_on_canvas.user_click] = 1
        for i in range(len(self.image_on_canvas.user_click)):
            naiveAdd[self.image_on_canvas.user_click[i]] = 1
        # save_path = self.posesavepath + self.image_name.split('/')[-1][:-4] + '_naiveAdd.png'
        # cv2.imwrite(save_path, naiveAdd*255)

        pred_af_naiveAdd = pred + naiveAdd
        pred_af_naiveAdd[np.where(pred_af_naiveAdd > 1)] = 1
        self.pred_final = pred_af_naiveAdd
        self.flag_final = 1
        self.image_on_canvas.user_click = []

    def _naiveEraser(self):
        naiveRadius = self.state['click_radius'].get()
        naiveErase = generate_interaction_maps([self.image_on_canvas.user_click], self.image_on_canvas.image_shape,
                                        interaction='scribbles', encoding='scribbles',
                                        radius=naiveRadius, gausigma=0.04, random_range=(-2, 2), op_way='middle', num_points=1)
        # save_path = self.neg_save_path + self.image_name.split('/')[-1][:-4] + '_naiveErase.png'
        # cv2.imwrite(save_path, naiveErase)
        pred = self.pred_final
        pred_af_naiveErase = pred*255 - naiveErase
        pred_af_naiveErase[np.where(pred_af_naiveErase < 0)] = 0
        self.pred_final = pred_af_naiveErase / 255.
        self.flag_final = 1
        self.image_on_canvas.user_click = []

    def _saveFinalEdgemap(self):
        save_path = self.interactive_res_path + self.image_name.split('/')[-1][:-4] + '_finalep.png'
        cv2.imwrite(save_path, self.controller.final_pred)

        # under a threshold
        save_path = self.interactive_res_path + self.image_name.split('/')[-1][:-4] + '_ob.png'
        cv2.imwrite(save_path, self.pred_final * 255)

    def _nms(self):
        # save to pre dir with nms
        coarse_ep = self.controller.final_pred / 255
        img_ob = np.array(coarse_ep.reshape(coarse_ep.shape[0], coarse_ep.shape[1])).astype(np.float64)
        obnms = nms_process_one_image(img_ob, None, False) / 255
        save_path = self.pre_save_path + self.image_name.split('/')[-1][:-4] + "_pynms.png"
        imageio.imwrite(save_path, obnms*255)
        self.nms_ep = obnms


    def _showVis(self):
        
        self._nms()

        if self.flag_final == 0:
            # threshold = self.state['prob_thresh'].get()
            # pred = (self.controller.final_pred > threshold).astype(int) 

            threshold = self.state['prob_thresh'].get()
            print()
            print(f"Current edge probobility thresholds is {threshold} !")
            print()
            pred = (self.nms_ep > threshold).astype(int)  # 0 or 1
            self.pred_final = pred 

        else:
            pred = self.pred_final

        if len(pred.shape) == 3:
            pred = pred.reshape((pred.shape[0], pred.shape[1]))
        image = cv2.cvtColor(cv2.imread(self.filename), cv2.COLOR_BGR2RGB)
        vis = draw_with_blend_and_clicks(image, pred, alpha=self.state['alpha_blend'].get())
        self.load_mask_k(vis)

    def _loadExistingOne(self):
        # should put existing edge in place
        img_name = self.filename.split('/')[-1].split('.')[0]
        path_existingEdgeMap = self.existing_pre_dir + img_name + '_ob.png'
        # default, don't has nms, or can use matlab do nms first then send to the local machine
        # 0-255 edge probability map or (1-p)
        p = cv2.imread(path_existingEdgeMap, cv2.IMREAD_GRAYSCALE)  # / 255.
        print()
        print("Successfully load the exsiting edge probobility map at ", path_existingEdgeMap)
        print()
        self.controller.final_pred = 255 - p
        save_path = self.pre_save_path + self.image_name.split('/')[-1][:-4] + "_pre.png"
        imageio.imwrite(save_path, self.controller.final_pred)


        
        
