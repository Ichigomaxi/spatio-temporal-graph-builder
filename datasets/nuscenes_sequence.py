

class NuscenesMOTSeqProcessor:
    """
    Class to process detections files coming from different mot_seqs.
    Main method is process_detections. It does the following:
    - Loads a DataFrameWSeqInfo (~pd.DataFrame) from a  detections file (self.det_df) via a the 'det_df_loader' func
    corresponding to the sequence type (mapped via _SEQ_TYPES)
    - Adds Sequence Info to the df (fps, img size, moving/static camera, etc.) as an additional attribute (_get_det_df)
    - If GT is available, assigns GT identities to the detected boxes via bipartite matching (_assign_gt)
    - Stores the df on disk (_store_det_df)
    - If required, precomputes CNN embeddings for every detected box and stores them on disk (_store_embeddings)

    The stored information assumes that each MOT sequence has its own directory. Inside it all processed data is
    stored as follows:
        +-- <Sequence name>
        |   +-- processed_data
        |       +-- det
        |           +-- <dataset_params['det_file_name']>.pkl # pd.DataFrame with processed detections and metainfo
        |       +-- embeddings
        |           +-- <dataset_params['det_file_name']> # Precomputed embeddings for a set of detections
        |               +-- <CNN Name >
        |                   +-- {frame1}.jpg
        |                   ...
        |                   +-- {frameN}.jpg
    """
    def __init__(self, dataset_path, seq_name, dataset_params, cnn_model = None, logger = None):
        self.seq_name = seq_name
        self.dataset_path = dataset_path
        self.seq_type = _SEQ_TYPES[seq_name]

        self.det_df_loader = _SEQ_TYPE_DETS_DF_LOADER[self.seq_type]
        self.dataset_params = dataset_params

        self.cnn_model = cnn_model

        self.logger = logger

    def load_or_process_detections(self):
        """
        Tries to load a set of processed detections if it's safe to do so. otherwise, it processes them and stores the
        result
        """
        # Check if the processed detections file already exists.
        seq_path = osp.join(self.dataset_path, self.seq_name)
        det_file_to_use = self.dataset_params['det_file_name'] if not self.seq_name.endswith('GT') else 'gt'
        seq_det_df_path = osp.join(seq_path, 'processed_data/det', det_file_to_use + '.pkl')

        # If loading precomputed embeddings, check if embeddings have already been stored (otherwise, we need to process dets again)
        node_embeds_path = osp.join(seq_path, 'processed_data/embeddings', det_file_to_use, self.dataset_params['node_embeddings_dir'])
        reid_embeds_path = osp.join(seq_path, 'processed_data/embeddings', det_file_to_use, self.dataset_params['reid_embeddings_dir'])
        try:
            num_frames = len(pd.read_pickle(seq_det_df_path)['frame'].unique())
            processed_dets_exist = True
        except:
            num_frames = -1
            processed_dets_exist = False

        embeds_ok = osp.exists(node_embeds_path) and len(os.listdir(node_embeds_path)) ==num_frames
        embeds_ok = embeds_ok and osp.exists(reid_embeds_path) and len(os.listdir(reid_embeds_path)) == num_frames
        embeds_ok = embeds_ok or not self.dataset_params['precomputed_embeddings']

        if processed_dets_exist and embeds_ok and not self.dataset_params['overwrite_processed_data']:
            print(f"Loading processed dets for sequence {self.seq_name} from {seq_det_df_path}")
            seq_det_df = pd.read_pickle(seq_det_df_path).reset_index().sort_values(by=['frame', 'detection_id'])

        else:
            print(f'Detections for sequence {self.seq_name} need to be processed. Starting processing')
            seq_det_df = self.process_detections()

        seq_det_df.seq_info_dict['seq_path'] = seq_path

        return seq_det_df