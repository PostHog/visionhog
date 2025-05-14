"""create stream chunks table

Revision ID: create_stream_chunks_table
Revises: b5bb0baa9d01
Create Date: 2024-05-14 16:30:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'create_stream_chunks_table'
down_revision = 'b5bb0baa9d01'
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Create stream_chunks table
    op.create_table(
        'stream_chunks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('stream_id', sa.Integer(), nullable=False),
        sa.Column('team_id', sa.String(), nullable=False),
        sa.Column('s3_video_key', sa.String(), nullable=False),
        sa.Column('s3_analysis_key', sa.String(), nullable=True),
        sa.Column('clip_name', sa.String(), nullable=False),
        sa.Column('processed_at', sa.DateTime(), nullable=False),
        sa.Column('processing_time', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('analysis_json', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['stream_id'], ['streams.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create index on stream_id and team_id for faster lookups
    op.create_index('ix_stream_chunks_stream_id', 'stream_chunks', ['stream_id'])
    op.create_index('ix_stream_chunks_team_id', 'stream_chunks', ['team_id'])

def downgrade() -> None:
    # Drop stream_chunks table
    op.drop_table('stream_chunks')