"""
[Model Forward function : This function get the output from the model every batch]
"""

def model_forward(model, args,loss_obj, batch):
    txt_left, segment_left, mask_left, \
    img_left,txt_right, segment_right, \
    mask_right, img_right,neg_txt_right, \
    neg_segment_right, neg_mask_right, neg_img_right, tgt = batch
    
    if args.freeze_img is True and args.freeze_txt is True:
        raise ValueError("Invalid input, cannot freeze image and text simultaneously.")
    
    for param in model.enc.img_encoder.parameters():
        param.requires_grad = not args.freeze_img
    for param in model.enc.encoder.parameters():
        param.requires_grad = not args.freeze_txt

    txt_left, img_left = txt_left.cuda(), img_left.cuda()
    txt_right, img_right = txt_right.cuda(), img_right.cuda()
    mask_left, segment_left = mask_left.cuda(), segment_left.cuda()
    mask_right, segment_right = mask_right.cuda(), segment_right.cuda()
    neg_txt_right,neg_mask_right,neg_segment_right,neg_img_right = neg_txt_right.cuda(), neg_mask_right.cuda(), neg_segment_right.cuda(),neg_img_right.cuda()
    
    out_l, out_r, out_neg_r = model(txt_left,txt_right,mask_left,
                                    mask_right,segment_left,segment_right,
                                    img_left,img_right, neg_txt_right,neg_mask_right,neg_segment_right,neg_img_right)
    
    tgt = tgt.cuda()
    res,loss = loss_obj(out_l,out_r, tgt)
    return loss,res,tgt

def model_forward_feat(model, args, batch):
    #! Feature extraction function
    txt, segment, mask, img, tgt = batch
    
    if args.freeze_img is True and args.freeze_txt is True:
        raise ValueError("Invalid input, cannot freeze image and text simultaneously.")
    
    for param in model.enc.img_encoder.parameters():
        param.requires_grad = not args.freeze_img
    for param in model.enc.encoder.parameters():
        param.requires_grad = not args.freeze_txt
    
    txt, img = txt.cuda(), img.cuda()
    mask, segment = mask.cuda(), segment.cuda()
    tgt = tgt.cuda()
    out = model(txt, mask, segment, img)
    
    return out, tgt