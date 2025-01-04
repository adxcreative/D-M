import argparse
import os.path


def parse_args():
    parser = argparse.ArgumentParser()

    # gpu & rank
    parser.add_argument( "--local_device_rank", type=int, default=0)
    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    # SFX_Moment ******************************************************************************************************
    # sfx
    parser.add_argument("--sfx_path", type=str, default='SFX_Moment/sfx/sfx_info.csv')
    parser.add_argument("--sfx_text_feat_path", type=str, default='SFX_Moment/sfx_feat/text')
    parser.add_argument("--sfx_ast_feat_path", type=str, default='SFX_Moment/sfx_feat/audio')
    # key moment
    parser.add_argument("--train_key_moment_path", type=str, default='SFX_Moment/video/train_keymoment.csv')
    parser.add_argument("--val_key_moment_path", type=str, default='SFX_Moment/video/val_keymoment.csv')
    parser.add_argument("--test_key_moment_path", type=str, default='SFX_Moment/video/test_keymoment.csv')
    # video
    parser.add_argument("--train_video_name_txt", type=str, default='SFX_Moment/train_video.txt')
    parser.add_argument("--train_tts_base", type=str, default='SFX_Moment/video/tts/train')
    parser.add_argument("--train_asr_base", type=str, default='SFX_Moment/video/asr/train')
    parser.add_argument("--train_frame_feat_base", type=str, default='SFX_Moment/video_feat/frame/train')
    parser.add_argument("--train_tts_feat_base", type=str, default='SFX_Moment/video_feat/tts/train')
    parser.add_argument("--train_asr_feat_base", type=str, default='SFX_Moment/video_feat/asr/train')

    parser.add_argument("--val_video_name_txt", type=str, default='SFX_Moment/val_video.txt')
    parser.add_argument("--val_tts_base", type=str, default='SFX_Moment/video/tts/val')
    parser.add_argument("--val_asr_base", type=str, default='SFX_Moment/video/asr/val')
    parser.add_argument("--val_frame_feat_base", type=str, default='SFX_Moment/video_feat/frame/val')
    parser.add_argument("--val_tts_feat_base", type=str, default='SFX_Moment/video_feat/tts/val')
    parser.add_argument("--val_asr_feat_base", type=str, default='SFX_Moment/video_feat/asr/val')

    parser.add_argument("--test_video_name_txt", type=str, default='SFX_Moment/test_video.txt')
    parser.add_argument("--test_tts_base", type=str, default='SFX_Moment/video/tts/test')
    parser.add_argument("--test_asr_base", type=str, default='SFX_Moment/video/asr/test')
    parser.add_argument("--test_frame_feat_base", type=str, default='SFX_Moment/video_feat/frame/test')
    parser.add_argument("--test_tts_feat_base", type=str, default='SFX_Moment/video_feat/tts/test')
    parser.add_argument("--test_asr_feat_base", type=str, default='SFX_Moment/video_feat/asr/test')

    parser.add_argument("--notext_np", type=str, default='SFX_Moment/video_feat/text_sp.npy')

    # HungarianMatcher *************************************************************************************************
    parser.add_argument("--HM_match", type=float, default=1)
    parser.add_argument("--HM_l1", type=float, default=1)
    parser.add_argument("--HM_giou", type=float, default=1)

    # loss ***********************************************************************************************************
    # pretrain
    parser.add_argument("--pretrain_msm_fore", type=float, default=1)
    parser.add_argument("--pretrain_msm_back", type=float, default=1)

    # train
    parser.add_argument("--fore_match", type=float, default=1)
    parser.add_argument("--l1", type=float, default=1)
    parser.add_argument("--giou", type=float, default=1)
    parser.add_argument("--back_match", type=float, default=1)

    # model ***********************************************************************************************************
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--nega_num", type=int, default=50)
    parser.add_argument( "--query_num", type=int, default=10)
    parser.add_argument("--limit_frame_num", type=int, default=12)

    parser.add_argument("--frame_dim", type=int, default=512)
    parser.add_argument("--text_dim", type=int, default=512)
    parser.add_argument("--ast_dim", type=int, default=1024)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--encoder_layer_num", type=int, default=6)
    parser.add_argument("--decoder_layer_num", type=int, default=6)
    parser.add_argument("--head_num", type=int, default=8)
    parser.add_argument("--att_dim", type=int, default=32)
    parser.add_argument("--att_dropout", type=float, default=0.1)
    parser.add_argument("--ffn_dropout", type=float, default=0.1)

    # lr **************************************************************************************************************
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=1e-8, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.001, help="Weight decay.")
    parser.add_argument("--warmup", type=int, default=100, help="Number of steps to warmup for.")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=3000)
    parser.add_argument("--tans_start_epoch", type=int, default=1000)
    parser.add_argument("--accum_freq", type=int, default=1)
    parser.add_argument("--nms", type=float, default=0.3)

    # save ************************************************************************************************************
    parser.add_argument("--save_period", type=int, default=100)
    parser.add_argument("--pretrain_loss_save_path", type=str, default='save/loss/pretrain_loss.npy')
    parser.add_argument("--pretrain_valrank1_save_path", type=str, default='save/loss/pretrain_valrank1.npy')
    parser.add_argument("--pretrain_model_base", type=str, default='save/pretrain_model')
    parser.add_argument("--pretrain_model_path", type=str, default='save/pretrain.pth')
    parser.add_argument("--train_loss_save_path", type=str, default='save/loss/pretrain_loss.npy')
    parser.add_argument("--train_valvdsfx_save_path", type=str, default='save/loss/train_valvdsfx.npy')
    parser.add_argument("--train_model_base", type=str, default='save/train_model')
    parser.add_argument("--train_model_path", type=str, default='save/train.pth')

    args = parser.parse_args()

    return args
